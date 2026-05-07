"""
Groq API client for LLM chat (streaming) and Whisper STT.
Uses LLaMA 3.3 70B Versatile for chat completions with streaming,
and Whisper Large V3 for speech-to-text transcription.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

from groq import AsyncGroq

logger = logging.getLogger(__name__)

# Model constants
CHAT_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3"

# Client singleton
_client: Optional[AsyncGroq] = None


def get_groq_client() -> AsyncGroq:
    """
    Get or create an async Groq client singleton.

    Returns:
        A configured AsyncGroq client.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY must be set in environment variables")

    _client = AsyncGroq(api_key=api_key)
    return _client


async def stream_chat_completion(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """
    Stream a chat completion from Groq LLaMA 3.3 70B.

    Yields individual text tokens as they arrive from the API.
    Uses streaming mode — never waits for the full response.

    Args:
        system_prompt: The system prompt with RAG context.
        user_message: The user's question.
        temperature: Sampling temperature (lower = more focused).
        max_tokens: Maximum tokens to generate.

    Yields:
        Individual text tokens (strings).

    Raises:
        Exception: If the Groq API call fails.
    """
    client = get_groq_client()

    try:
        stream = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

    except Exception as exc:
        logger.error("Groq streaming chat error: %s", exc)
        raise


async def transcribe_audio(
    audio_file_path: str,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe an audio file using Groq's Whisper Large V3.

    Args:
        audio_file_path: Path to the audio file (mp3, wav, m4a, etc.).
        language: Optional ISO-639-1 language code (e.g., 'en').

    Returns:
        The transcribed text.

    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        Exception: If the Groq API call fails.
    """
    file_path = Path(audio_file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    client = get_groq_client()

    try:
        with open(file_path, "rb") as audio_file:
            transcription_params = {
                "model": WHISPER_MODEL,
                "file": audio_file,
                "response_format": "text",
            }
            if language:
                transcription_params["language"] = language

            transcription = await client.audio.transcriptions.create(
                **transcription_params
            )

        transcript_text = str(transcription).strip()
        logger.info(
            "Transcribed %s: %d characters",
            file_path.name,
            len(transcript_text),
        )
        return transcript_text

    except Exception as exc:
        logger.error("Groq Whisper transcription error: %s", exc)
        raise


async def generate_tags(text: str, max_tags: int = 5) -> list[str]:
    """
    Auto-generate descriptive tags for a text using Groq LLaMA.

    Uses a streaming call but collects the full response since tags
    are short and need to be parsed as a complete list.

    Args:
        text: The text to generate tags for (first 1000 chars used).
        max_tags: Maximum number of tags to generate.

    Returns:
        A list of tag strings.
    """
    client = get_groq_client()

    # Use a truncated version for tagging efficiency
    truncated = text[:1000] if len(text) > 1000 else text

    try:
        stream = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Generate exactly {max_tags} short, descriptive tags for the following text. "
                        "Return ONLY the tags as a comma-separated list. No explanations, no numbering. "
                        "Example output: machine learning, neural networks, deep learning, AI, computer vision"
                    ),
                },
                {"role": "user", "content": truncated},
            ],
            temperature=0.2,
            max_tokens=100,
            stream=True,
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_response += delta.content

        # Parse comma-separated tags
        tags = [
            tag.strip().lower()
            for tag in full_response.split(",")
            if tag.strip()
        ]

        # Limit to max_tags
        tags = tags[:max_tags]
        logger.info("Generated tags: %s", tags)
        return tags

    except Exception as exc:
        logger.warning("Tag generation failed (non-critical): %s", exc)
        return []


async def check_groq_health() -> bool:
    """
    Check if the Groq API is accessible.

    Returns:
        True if Groq responds, False otherwise.
    """
    try:
        client = get_groq_client()
        # Quick non-streaming health check
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            stream=False,
        )
        return bool(response.choices)
    except Exception as exc:
        logger.error("Groq health check failed: %s", exc)
        return False
