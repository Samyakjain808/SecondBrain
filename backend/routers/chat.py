"""
RAG Chat router — the core intelligence of the Second Brain.
Embeds the user query, searches for relevant chunks via pgvector,
optionally re-ranks with Cohere, builds a context-rich prompt,
and streams Groq LLaMA's response as Server-Sent Events.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import cohere
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from models.schemas import ChatRequest, SourceChunk
from services.embedder import generate_embedding
from services.groq_client import stream_chat_completion
from services.vector_store import (
    get_knowledge_item_titles,
    similarity_search,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Cohere client singleton
_cohere_client: Optional[cohere.ClientV2] = None


def _get_cohere_client() -> Optional[cohere.ClientV2]:
    """Get or create a Cohere client for re-ranking."""
    global _cohere_client
    if _cohere_client is not None:
        return _cohere_client

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        logger.warning("COHERE_API_KEY not set — re-ranking disabled")
        return None

    _cohere_client = cohere.ClientV2(api_key=api_key)
    return _cohere_client


async def _rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Re-rank retrieved chunks using Cohere Rerank API.

    Takes the initial retrieval results and re-orders them by
    relevance using a cross-encoder model for higher accuracy.

    Args:
        query: The user's original question.
        chunks: List of chunk dicts from similarity search.
        top_n: Number of top results to keep after re-ranking.

    Returns:
        Re-ranked list of chunk dicts (top_n items).
    """
    client = _get_cohere_client()
    if not client or not chunks:
        return chunks[:top_n]

    try:
        documents = [chunk["content"] for chunk in chunks]

        response = client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
            top_n=min(top_n, len(chunks)),
        )

        # Re-order chunks based on Cohere's ranking
        reranked: list[dict] = []
        for result in response.results:
            idx = result.index
            chunk = chunks[idx].copy()
            chunk["rerank_score"] = result.relevance_score
            reranked.append(chunk)

        logger.info(
            "Re-ranked %d chunks → top %d (scores: %.3f to %.3f)",
            len(chunks),
            len(reranked),
            reranked[0]["rerank_score"] if reranked else 0,
            reranked[-1]["rerank_score"] if reranked else 0,
        )

        return reranked

    except Exception as exc:
        logger.warning("Cohere re-ranking failed (falling back): %s", exc)
        return chunks[:top_n]


def _build_rag_prompt(chunks: list[dict], source_info: dict[str, dict]) -> str:
    """
    Build the system prompt with retrieved context chunks.

    Creates a structured prompt that instructs the LLM to answer
    based on the provided context and cite its sources.

    Args:
        chunks: List of relevant chunk dicts.
        source_info: Dict mapping knowledge_item_id to {title, source_type}.

    Returns:
        Complete system prompt string.
    """
    context_parts: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        item_id = chunk.get("knowledge_item_id", "unknown")
        info = source_info.get(item_id, {})
        title = info.get("title", "Unknown Source")
        source_type = info.get("source_type", "unknown")

        context_parts.append(
            f"[Source {i}: {title} ({source_type})]\n{chunk['content']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    system_prompt = f"""You are an AI assistant for a personal knowledge base called "Second Brain". Your role is to answer questions based ONLY on the user's stored knowledge.

## Instructions:
1. Answer the user's question using ONLY the context provided below.
2. If the context doesn't contain enough information, say so honestly — never make up facts.
3. Cite your sources by referring to them as [Source N] where N is the source number.
4. Be concise but thorough. Use bullet points for lists.
5. If multiple sources support an answer, mention all of them.
6. Maintain a helpful, knowledgeable tone.

## User's Knowledge Base Context:

{context_block}

## Important:
- Only use information from the sources above.
- Always cite which source(s) you're drawing from.
- If you're unsure, say "Based on your stored knowledge, I don't have enough information to answer this definitively."
"""

    return system_prompt


# ──────────────────────────────────────────────
# POST /chat
# ──────────────────────────────────────────────

@router.post("")
async def chat(request: ChatRequest):
    """
    RAG-powered chat endpoint.

    1. Embeds the user's query
    2. Searches for top-K relevant chunks via pgvector
    3. Optionally re-ranks with Cohere (top-N)
    4. Builds a context-rich system prompt
    5. Streams Groq LLaMA's response as Server-Sent Events
    6. Sends source citations as a final SSE event

    SSE Event Types:
    - "token": A single text token from the LLM response
    - "sources": JSON array of source chunks used (sent once)
    - "done": Stream complete signal
    - "error": Error message
    """
    logger.info(
        "Chat query from user %s: '%s' (top_k=%d, rerank=%s)",
        request.user_id,
        request.query[:100],
        request.top_k,
        request.use_reranking,
    )

    try:
        # Step 1: Embed the query
        query_embedding = await generate_embedding(request.query)

        # Step 2: Similarity search
        raw_chunks = await similarity_search(
            query_embedding=query_embedding,
            user_id=request.user_id,
            top_k=request.top_k,
        )

        if not raw_chunks:
            async def no_results_stream():
                yield {
                    "event": "token",
                    "data": "I couldn't find any relevant information in your knowledge base. Try uploading some content first!",
                }
                yield {"event": "sources", "data": "[]"}
                yield {"event": "done", "data": ""}

            return EventSourceResponse(no_results_stream())

        # Step 3: Re-rank with Cohere (optional)
        if request.use_reranking:
            chunks = await _rerank_chunks(
                query=request.query,
                chunks=raw_chunks,
                top_n=request.top_n,
            )
        else:
            chunks = raw_chunks[:request.top_n]

        # Step 4: Get source titles for citation
        item_ids = list({c["knowledge_item_id"] for c in chunks})
        source_info = await get_knowledge_item_titles(item_ids)

        # Build source chunks for the response
        source_chunks: list[dict] = []
        for chunk in chunks:
            item_id = chunk.get("knowledge_item_id", "")
            info = source_info.get(item_id, {})
            source_chunks.append(
                SourceChunk(
                    chunk_id=chunk.get("id", ""),
                    content=chunk.get("content", ""),
                    knowledge_item_id=item_id,
                    similarity=chunk.get("similarity", 0.0),
                    title=info.get("title"),
                    source_type=info.get("source_type"),
                ).model_dump()
            )

        # Step 5: Build RAG prompt
        system_prompt = _build_rag_prompt(chunks, source_info)

        # Step 6: Stream the response
        async def event_stream():
            try:
                # Stream tokens from Groq
                async for token in stream_chat_completion(
                    system_prompt=system_prompt,
                    user_message=request.query,
                ):
                    yield {"event": "token", "data": token}

                # Send sources after the response completes
                yield {
                    "event": "sources",
                    "data": json.dumps(source_chunks),
                }

                yield {"event": "done", "data": ""}

            except Exception as exc:
                logger.error("Chat streaming error: %s", exc)
                yield {
                    "event": "error",
                    "data": f"Streaming error: {exc}",
                }

        return EventSourceResponse(event_stream())

    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {exc}",
        )
