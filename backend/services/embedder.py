"""
Embedding service using nomic-embed-text via Ollama.
Generates 768-dimensional vector embeddings for text chunks and queries,
enabling semantic similarity search in the vector database.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768
TIMEOUT_SECONDS = 60.0


async def generate_embedding(text: str) -> list[float]:
    """
    Generate a 768-dimensional embedding for a single text string.

    Args:
        text: The text to embed.

    Returns:
        A list of 768 floats representing the embedding vector.

    Raises:
        ValueError: If text is empty.
        httpx.HTTPStatusError: If the Ollama API returns an error.
        ConnectionError: If Ollama server is unreachable.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text.strip(),
                },
            )
            response.raise_for_status()
            data = response.json()

            # Ollama returns {"embeddings": [[...vectors...]]}
            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) == 0:
                raise ValueError(f"Ollama returned empty embeddings for model {EMBEDDING_MODEL}")

            embedding = embeddings[0]

            if len(embedding) != EMBEDDING_DIMENSION:
                logger.warning(
                    "Expected %d dimensions, got %d",
                    EMBEDDING_DIMENSION,
                    len(embedding),
                )

            return embedding

    except httpx.ConnectError as exc:
        logger.error("Cannot connect to Ollama at %s: %s", OLLAMA_BASE_URL, exc)
        raise ConnectionError(
            f"Ollama server is not running at {OLLAMA_BASE_URL}. "
            f"Please start it with: ollama serve"
        ) from exc


async def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Processes each text sequentially to avoid overwhelming the local Ollama server.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each is a list of 768 floats).

    Raises:
        ValueError: If the texts list is empty.
    """
    if not texts:
        raise ValueError("Cannot embed empty list of texts")

    embeddings: list[list[float]] = []
    for i, text in enumerate(texts):
        logger.debug("Embedding chunk %d/%d", i + 1, len(texts))
        embedding = await generate_embedding(text)
        embeddings.append(embedding)

    logger.info("Generated %d embeddings", len(embeddings))
    return embeddings


async def check_ollama_health() -> bool:
    """
    Check if the Ollama server is running and the embedding model is available.

    Returns:
        True if Ollama is healthy and the model is pulled, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check server is up
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()

            # Check if nomic-embed-text model is available
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]

            is_available = any(EMBEDDING_MODEL in name for name in model_names)

            if not is_available:
                logger.warning(
                    "Model '%s' not found in Ollama. Available models: %s. "
                    "Run: ollama pull %s",
                    EMBEDDING_MODEL,
                    model_names,
                    EMBEDDING_MODEL,
                )

            return is_available

    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        logger.error("Ollama health check failed: %s", exc)
        return False
