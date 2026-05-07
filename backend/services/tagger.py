"""
Auto-tagging service using Groq LLaMA 3.3 70B.
Generates 3-5 descriptive tags for any knowledge item based on its content,
enabling tag-based filtering and discovery in the dashboard.
"""

from __future__ import annotations

import logging

from services.groq_client import generate_tags
from services.vector_store import update_knowledge_item_tags

logger = logging.getLogger(__name__)


async def auto_tag_knowledge_item(
    item_id: str,
    text: str,
    max_tags: int = 5,
) -> list[str]:
    """
    Auto-generate tags for a knowledge item and persist them.

    Takes the first 1000 characters of the text, sends them to Groq
    to generate descriptive tags, then updates the knowledge item
    in the database.

    Args:
        item_id: UUID of the knowledge item to tag.
        text: The full text content to generate tags from.
        max_tags: Maximum number of tags (default: 5).

    Returns:
        List of generated tag strings.
    """
    if not text or not text.strip():
        logger.warning("Cannot tag item %s — empty text", item_id)
        return []

    try:
        # Generate tags via Groq LLaMA
        tags = await generate_tags(text=text, max_tags=max_tags)

        if tags:
            # Persist tags to the database
            await update_knowledge_item_tags(item_id=item_id, tags=tags)
            logger.info("Tagged item %s with: %s", item_id, tags)
        else:
            logger.warning("No tags generated for item %s", item_id)

        return tags

    except Exception as exc:
        # Tagging is non-critical — log and return empty
        logger.warning(
            "Auto-tagging failed for item %s (non-critical): %s",
            item_id,
            exc,
        )
        return []
