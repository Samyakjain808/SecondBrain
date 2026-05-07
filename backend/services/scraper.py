"""
Web scraping service using Firecrawl SDK.
Scrapes any URL and returns clean, markdown-formatted text
suitable for chunking and embedding.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from firecrawl import FirecrawlApp

logger = logging.getLogger(__name__)

# Client singleton
_client: Optional[FirecrawlApp] = None


def get_firecrawl_client() -> FirecrawlApp:
    """
    Get or create a Firecrawl client singleton.

    Returns:
        A configured FirecrawlApp client.

    Raises:
        ValueError: If FIRECRAWL_API_KEY is not set.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY must be set in environment variables")

    _client = FirecrawlApp(api_key=api_key)
    return _client


async def scrape_url(url: str) -> dict[str, str]:
    """
    Scrape a web page and extract its content as clean text.

    Uses Firecrawl to handle JavaScript rendering, popups, and
    other dynamic content. Returns markdown-formatted text.

    Args:
        url: The URL to scrape.

    Returns:
        A dict with keys:
            - title (str): Page title
            - content (str): Clean text content in markdown format
            - source_url (str): The original URL

    Raises:
        ValueError: If the URL is empty or scraping returns no content.
        Exception: If Firecrawl API call fails.
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    logger.info("Scraping URL: %s", url)

    try:
        client = get_firecrawl_client()

        # Scrape the page — Firecrawl handles JS rendering automatically
        result = client.scrape_url(
            url=url.strip(),
            params={
                "formats": ["markdown"],
            },
        )

        if not result:
            raise ValueError(f"Firecrawl returned empty result for {url}")

        # Extract content from the result
        content = ""
        title = ""

        if isinstance(result, dict):
            # Try to get markdown content
            content = result.get("markdown", "")
            if not content:
                content = result.get("content", "")

            # Try to get metadata for title
            metadata = result.get("metadata", {})
            if isinstance(metadata, dict):
                title = metadata.get("title", "")
                if not title:
                    title = metadata.get("og:title", "")
        elif isinstance(result, str):
            content = result

        if not content or not content.strip():
            raise ValueError(
                f"No text content could be extracted from {url}. "
                "The page may be empty, behind a login, or blocked."
            )

        # Fallback title from URL if none found
        if not title:
            title = _extract_title_from_url(url)

        logger.info(
            "Scraped %s: '%s' (%d characters)",
            url,
            title[:50],
            len(content),
        )

        return {
            "title": title,
            "content": content.strip(),
            "source_url": url,
        }

    except ValueError:
        raise
    except Exception as exc:
        logger.error("Failed to scrape %s: %s", url, exc)
        raise ValueError(f"Failed to scrape URL: {exc}") from exc


def _extract_title_from_url(url: str) -> str:
    """
    Extract a readable title from a URL as a fallback.

    Args:
        url: The URL string.

    Returns:
        A human-readable title derived from the URL path.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    # Use the hostname + last path segment
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]

    if path_parts:
        last_segment = path_parts[-1]
        # Clean up the segment
        title = last_segment.replace("-", " ").replace("_", " ").title()
        return f"{parsed.hostname} — {title}"

    return parsed.hostname or url[:50]
