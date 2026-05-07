"""
Knowledge management router — CRUD and semantic search for knowledge items.
Provides endpoints to list, delete, and search the user's knowledge base
without triggering the full RAG chat pipeline.
"""

from __future__ import annotations

import logging
from collections import Counter

from fastapi import APIRouter, HTTPException, Query

from models.schemas import (
    DashboardStats,
    KnowledgeDeleteResponse,
    KnowledgeItem,
    KnowledgeListResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeSearchResult,
)
from services.embedder import generate_embedding
from services.vector_store import (
    delete_knowledge_item,
    get_chunk_count,
    get_knowledge_item,
    get_knowledge_item_titles,
    get_knowledge_items,
    similarity_search,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ──────────────────────────────────────────────
# GET /knowledge — List all knowledge items
# ──────────────────────────────────────────────

@router.get("", response_model=KnowledgeListResponse)
async def list_knowledge(
    user_id: str = Query(..., description="UUID of the authenticated user"),
) -> KnowledgeListResponse:
    """
    List all knowledge items for a user, ordered by creation date (newest first).

    Returns item metadata including title, source type, tags, and chunk count.
    """
    logger.info("Listing knowledge items for user %s", user_id)

    try:
        items = await get_knowledge_items(user_id=user_id)

        knowledge_items = [
            KnowledgeItem(
                id=item["id"],
                user_id=item["user_id"],
                title=item["title"],
                source_type=item["source_type"],
                source_url=item.get("source_url"),
                raw_text=item.get("raw_text"),
                created_at=item.get("created_at"),
                tags=item.get("tags", []),
            )
            for item in items
        ]

        return KnowledgeListResponse(
            items=knowledge_items,
            total=len(knowledge_items),
        )

    except Exception as exc:
        logger.error("Failed to list knowledge items: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list knowledge items: {exc}",
        )


# ──────────────────────────────────────────────
# GET /knowledge/{item_id} — Get a single item
# ──────────────────────────────────────────────

@router.get("/{item_id}", response_model=KnowledgeItem)
async def get_single_knowledge_item(item_id: str) -> KnowledgeItem:
    """
    Get a single knowledge item by its UUID.

    Returns the full item including raw text content.
    """
    logger.info("Getting knowledge item %s", item_id)

    try:
        item = await get_knowledge_item(item_id=item_id)

        if not item:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge item {item_id} not found",
            )

        return KnowledgeItem(
            id=item["id"],
            user_id=item["user_id"],
            title=item["title"],
            source_type=item["source_type"],
            source_url=item.get("source_url"),
            raw_text=item.get("raw_text"),
            created_at=item.get("created_at"),
            tags=item.get("tags", []),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get knowledge item: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get knowledge item: {exc}",
        )


# ──────────────────────────────────────────────
# DELETE /knowledge/{item_id} — Delete an item
# ──────────────────────────────────────────────

@router.delete("/{item_id}", response_model=KnowledgeDeleteResponse)
async def delete_single_knowledge_item(item_id: str) -> KnowledgeDeleteResponse:
    """
    Delete a knowledge item and all its associated chunks.

    Chunks are automatically cascade-deleted by the database.
    """
    logger.info("Deleting knowledge item %s", item_id)

    try:
        deleted = await delete_knowledge_item(item_id=item_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge item {item_id} not found",
            )

        return KnowledgeDeleteResponse(
            deleted_id=item_id,
            message=f"Knowledge item {item_id} and all its chunks deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete knowledge item: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete knowledge item: {exc}",
        )


# ──────────────────────────────────────────────
# POST /knowledge/search — Semantic search
# ──────────────────────────────────────────────

@router.post("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """
    Semantic search across the user's knowledge base.

    Embeds the search query and finds the most similar chunks
    using cosine similarity — no chat, just search results.
    """
    logger.info(
        "Semantic search for user %s: '%s'",
        request.user_id,
        request.query[:100],
    )

    try:
        # Embed the search query
        query_embedding = await generate_embedding(request.query)

        # Run similarity search
        matches = await similarity_search(
            query_embedding=query_embedding,
            user_id=request.user_id,
            top_k=request.top_k,
        )

        if not matches:
            return KnowledgeSearchResponse(
                results=[],
                query=request.query,
            )

        # Get titles for the matching items
        item_ids = list({m["knowledge_item_id"] for m in matches})
        source_info = await get_knowledge_item_titles(item_ids)

        # Build search results
        results = [
            KnowledgeSearchResult(
                chunk_id=match["id"],
                content=match["content"],
                knowledge_item_id=match["knowledge_item_id"],
                title=source_info.get(match["knowledge_item_id"], {}).get("title"),
                source_type=source_info.get(match["knowledge_item_id"], {}).get("source_type"),
                similarity=match.get("similarity", 0.0),
            )
            for match in matches
        ]

        return KnowledgeSearchResponse(
            results=results,
            query=request.query,
        )

    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Semantic search failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}",
        )


# ──────────────────────────────────────────────
# GET /knowledge/stats — Dashboard statistics
# ──────────────────────────────────────────────

@router.get("/stats/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    user_id: str = Query(..., description="UUID of the authenticated user"),
) -> DashboardStats:
    """
    Get aggregated statistics for the user's dashboard.

    Returns total items, total chunks, breakdown by source type,
    and all unique tags.
    """
    logger.info("Getting dashboard stats for user %s", user_id)

    try:
        # Get all knowledge items
        items = await get_knowledge_items(user_id=user_id)

        # Get total chunk count
        total_chunks = await get_chunk_count(user_id=user_id)

        # Aggregate source types
        type_counts = Counter(item["source_type"] for item in items)

        # Collect all unique tags
        all_tags: set[str] = set()
        for item in items:
            tags = item.get("tags", [])
            if tags:
                all_tags.update(tags)

        return DashboardStats(
            total_items=len(items),
            total_chunks=total_chunks,
            sources_by_type=dict(type_counts),
            tags=sorted(all_tags),
        )

    except Exception as exc:
        logger.error("Failed to get dashboard stats: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dashboard stats: {exc}",
        )
