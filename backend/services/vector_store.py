"""
Vector store service using Supabase with pgvector.
Handles storing embedded chunks and running cosine similarity searches
against the user's knowledge base.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Supabase client singleton
_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """
    Get or create a Supabase client singleton.

    Returns:
        A configured Supabase client.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_SERVICE_KEY are not set.
    """
    global _client
    if _client is not None:
        return _client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables"
        )

    _client = create_client(url, key)
    return _client


async def create_knowledge_item(
    user_id: str,
    title: str,
    source_type: str,
    raw_text: str,
    source_url: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
    """
    Create a new knowledge item in the database.

    Args:
        user_id: UUID of the owning user.
        title: Title of the knowledge item.
        source_type: One of 'pdf', 'url', 'voice', 'text'.
        raw_text: Full raw text content.
        source_url: Optional source URL.
        tags: Optional list of tags.

    Returns:
        The UUID of the created knowledge item.

    Raises:
        Exception: If the database insert fails.
    """
    client = get_supabase_client()

    data = {
        "user_id": user_id,
        "title": title,
        "source_type": source_type,
        "raw_text": raw_text,
        "source_url": source_url,
        "tags": tags or [],
    }

    result = client.table("knowledge_items").insert(data).execute()

    if not result.data or len(result.data) == 0:
        raise Exception("Failed to create knowledge item — no data returned")

    item_id = result.data[0]["id"]
    logger.info("Created knowledge item %s: '%s'", item_id, title)
    return item_id


async def store_chunks(
    chunks_data: list[dict],
) -> int:
    """
    Store embedded chunks in the database.

    Args:
        chunks_data: List of dicts with keys:
            - knowledge_item_id (str)
            - user_id (str)
            - content (str)
            - embedding (list[float])
            - chunk_index (int)
            - token_count (int)

    Returns:
        Number of chunks successfully stored.

    Raises:
        Exception: If the database insert fails.
    """
    if not chunks_data:
        return 0

    client = get_supabase_client()

    # Supabase can handle batch inserts
    result = client.table("chunks").insert(chunks_data).execute()

    stored_count = len(result.data) if result.data else 0
    logger.info("Stored %d chunks", stored_count)
    return stored_count


async def similarity_search(
    query_embedding: list[float],
    user_id: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Perform cosine similarity search against the user's chunks.

    Uses the Supabase RPC function `match_chunks` which runs a
    vector similarity search using pgvector.

    Args:
        query_embedding: 768-dimensional query embedding vector.
        user_id: UUID of the user to search within.
        top_k: Number of top results to return.

    Returns:
        List of dicts with keys: id, content, knowledge_item_id, similarity.

    Raises:
        Exception: If the RPC call fails.
    """
    client = get_supabase_client()

    result = client.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "match_user_id": user_id,
            "match_count": top_k,
        },
    ).execute()

    matches = result.data if result.data else []
    logger.info(
        "Similarity search returned %d results for user %s",
        len(matches),
        user_id,
    )
    return matches


async def get_knowledge_items(user_id: str) -> list[dict]:
    """
    Get all knowledge items for a user, ordered by creation date.

    Args:
        user_id: UUID of the user.

    Returns:
        List of knowledge item dicts.
    """
    client = get_supabase_client()

    result = (
        client.table("knowledge_items")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return result.data if result.data else []


async def get_knowledge_item(item_id: str) -> Optional[dict]:
    """
    Get a single knowledge item by ID.

    Args:
        item_id: UUID of the knowledge item.

    Returns:
        Knowledge item dict, or None if not found.
    """
    client = get_supabase_client()

    result = (
        client.table("knowledge_items")
        .select("*")
        .eq("id", item_id)
        .maybe_single()
        .execute()
    )

    return result.data


async def delete_knowledge_item(item_id: str) -> bool:
    """
    Delete a knowledge item and all its associated chunks.
    Chunks are auto-deleted via cascade in the database schema.

    Args:
        item_id: UUID of the knowledge item to delete.

    Returns:
        True if the item was deleted, False otherwise.
    """
    client = get_supabase_client()

    result = (
        client.table("knowledge_items")
        .delete()
        .eq("id", item_id)
        .execute()
    )

    deleted = bool(result.data and len(result.data) > 0)
    if deleted:
        logger.info("Deleted knowledge item %s (chunks cascaded)", item_id)
    else:
        logger.warning("Knowledge item %s not found for deletion", item_id)

    return deleted


async def get_chunk_count(user_id: str) -> int:
    """
    Get the total number of chunks for a user.

    Args:
        user_id: UUID of the user.

    Returns:
        Total chunk count.
    """
    client = get_supabase_client()

    result = (
        client.table("chunks")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .execute()
    )

    return result.count if result.count is not None else 0


async def get_knowledge_item_titles(item_ids: list[str]) -> dict[str, dict]:
    """
    Get titles and source types for a list of knowledge item IDs.

    Args:
        item_ids: List of knowledge item UUIDs.

    Returns:
        Dict mapping item_id to {title, source_type}.
    """
    if not item_ids:
        return {}

    client = get_supabase_client()

    result = (
        client.table("knowledge_items")
        .select("id, title, source_type")
        .in_("id", item_ids)
        .execute()
    )

    return {
        item["id"]: {"title": item["title"], "source_type": item["source_type"]}
        for item in (result.data or [])
    }


async def update_knowledge_item_tags(item_id: str, tags: list[str]) -> None:
    """
    Update the tags for a knowledge item.

    Args:
        item_id: UUID of the knowledge item.
        tags: New list of tags.
    """
    client = get_supabase_client()

    client.table("knowledge_items").update({"tags": tags}).eq("id", item_id).execute()
    logger.info("Updated tags for item %s: %s", item_id, tags)


async def check_supabase_health() -> bool:
    """
    Check if the Supabase connection and tables are accessible.

    Returns:
        True if healthy, False otherwise.
    """
    try:
        client = get_supabase_client()
        # Try a simple query to verify connection
        client.table("knowledge_items").select("id").limit(1).execute()
        return True
    except Exception as exc:
        logger.error("Supabase health check failed: %s", exc)
        return False
