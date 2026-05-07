"""
Text chunking service using LangChain's RecursiveCharacterTextSplitter.
Splits ingested content into overlapping chunks of ~512 tokens for optimal
embedding and retrieval performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# tiktoken encoding for LLaMA-compatible token counting
ENCODING_NAME = "cl100k_base"
CHUNK_SIZE = 512          # target tokens per chunk
CHUNK_OVERLAP = 50        # overlap tokens between consecutive chunks


@dataclass
class TextChunk:
    """A single chunk of text with metadata."""
    content: str
    chunk_index: int
    token_count: int


def _get_token_counter() -> tiktoken.Encoding:
    """Get the tiktoken encoding instance."""
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    enc = _get_token_counter()
    return len(enc.encode(text))


def chunk_text(text: str) -> list[TextChunk]:
    """
    Split text into overlapping chunks optimized for embedding.

    Uses RecursiveCharacterTextSplitter with tiktoken-based length function
    to ensure each chunk is approximately 512 tokens with 50 tokens of overlap.

    Args:
        text: The raw text to split into chunks.

    Returns:
        A list of TextChunk objects with content, index, and token count.

    Raises:
        ValueError: If the input text is empty or None.
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty or whitespace-only text")

    enc = _get_token_counter()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda t: len(enc.encode(t)),
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        is_separator_regex=False,
    )

    raw_chunks = splitter.split_text(text.strip())

    chunks: list[TextChunk] = []
    for idx, chunk_content in enumerate(raw_chunks):
        token_count = len(enc.encode(chunk_content))
        chunks.append(
            TextChunk(
                content=chunk_content,
                chunk_index=idx,
                token_count=token_count,
            )
        )

    logger.info(
        "Chunked %d characters into %d chunks (avg %d tokens/chunk)",
        len(text),
        len(chunks),
        sum(c.token_count for c in chunks) // max(len(chunks), 1),
    )

    return chunks
