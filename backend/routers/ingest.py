"""
Ingestion router — handles uploading and processing all knowledge sources.
Four endpoints for PDF files, web URLs, voice recordings, and raw text.
Each endpoint: parses → chunks → embeds → stores in Supabase pgvector.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.schemas import IngestResponse, IngestTextRequest, IngestURLRequest, SourceType
from services.chunker import chunk_text
from services.embedder import generate_embeddings_batch
from services.groq_client import transcribe_audio
from services.scraper import scrape_url
from services.tagger import auto_tag_knowledge_item
from services.vector_store import create_knowledge_item, store_chunks

logger = logging.getLogger(__name__)

router = APIRouter()

# Max upload sizes
MAX_PDF_SIZE = 50 * 1024 * 1024    # 50 MB
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB (Groq Whisper limit)


# ──────────────────────────────────────────────
# Helper: Common ingestion pipeline
# ──────────────────────────────────────────────

async def _ingest_pipeline(
    user_id: str,
    title: str,
    source_type: SourceType,
    raw_text: str,
    source_url: str | None = None,
) -> IngestResponse:
    """
    Common pipeline shared by all ingestion endpoints.
    Chunks the text, generates embeddings, stores everything in Supabase,
    and auto-tags the knowledge item.

    Args:
        user_id: UUID of the authenticated user.
        title: Title for the knowledge item.
        source_type: Type of source (pdf, url, voice, text).
        raw_text: The full extracted text content.
        source_url: Optional URL source.

    Returns:
        IngestResponse with the created item details.

    Raises:
        HTTPException: If any step in the pipeline fails.
    """
    try:
        # Step 1: Create the knowledge item in the database
        item_id = await create_knowledge_item(
            user_id=user_id,
            title=title,
            source_type=source_type.value,
            raw_text=raw_text,
            source_url=source_url,
        )
        logger.info("Created knowledge item: %s", item_id)

        # Step 2: Chunk the text
        text_chunks = chunk_text(raw_text)
        logger.info("Created %d chunks from '%s'", len(text_chunks), title)

        # Step 3: Generate embeddings for all chunks
        chunk_contents = [c.content for c in text_chunks]
        embeddings = await generate_embeddings_batch(chunk_contents)
        logger.info("Generated %d embeddings", len(embeddings))

        # Step 4: Store chunks with embeddings in Supabase
        chunks_data = [
            {
                "knowledge_item_id": item_id,
                "user_id": user_id,
                "content": text_chunks[i].content,
                "embedding": embeddings[i],
                "chunk_index": text_chunks[i].chunk_index,
                "token_count": text_chunks[i].token_count,
            }
            for i in range(len(text_chunks))
        ]
        stored_count = await store_chunks(chunks_data)
        logger.info("Stored %d chunks in vector DB", stored_count)

        # Step 5: Auto-tag (non-blocking — failure doesn't break ingestion)
        tags = await auto_tag_knowledge_item(item_id=item_id, text=raw_text)

        return IngestResponse(
            knowledge_item_id=item_id,
            title=title,
            source_type=source_type,
            chunks_created=stored_count,
            tags=tags,
            message=f"Successfully ingested '{title}' — {stored_count} chunks created",
        )

    except ValueError as exc:
        logger.error("Ingestion validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except ConnectionError as exc:
        logger.error("Service connection error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Ingestion pipeline error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        )


# ──────────────────────────────────────────────
# POST /ingest/text
# ──────────────────────────────────────────────

@router.post("/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest) -> IngestResponse:
    """
    Ingest raw text as a knowledge item.

    Accepts plain text content, chunks it, generates embeddings,
    and stores everything in the vector database.
    """
    logger.info("Ingesting text: '%s' for user %s", request.title, request.user_id)

    return await _ingest_pipeline(
        user_id=request.user_id,
        title=request.title,
        source_type=SourceType.TEXT,
        raw_text=request.content,
    )


# ──────────────────────────────────────────────
# POST /ingest/pdf
# ──────────────────────────────────────────────

@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    title: str = Form(None, description="Optional title (defaults to filename)"),
    user_id: str = Form(..., description="UUID of the authenticated user"),
) -> IngestResponse:
    """
    Ingest a PDF file as a knowledge item.

    Extracts text from all pages using PyMuPDF, chunks it,
    generates embeddings, and stores in the vector database.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    # Validate file size
    content = await file.read()
    if len(content) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"PDF file is too large. Maximum size is {MAX_PDF_SIZE // (1024 * 1024)} MB.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="PDF file is empty.")

    # Use filename as title if not provided
    pdf_title = title or file.filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()

    logger.info("Ingesting PDF: '%s' (%d bytes) for user %s", pdf_title, len(content), user_id)

    # Extract text from PDF using PyMuPDF
    try:
        raw_text = _extract_pdf_text(content)
    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract text from PDF: {exc}",
        )

    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this PDF. It may be image-only or corrupted.",
        )

    return await _ingest_pipeline(
        user_id=user_id,
        title=pdf_title,
        source_type=SourceType.PDF,
        raw_text=raw_text,
    )


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        Concatenated text from all pages.

    Raises:
        Exception: If the PDF cannot be opened or parsed.
    """
    text_parts: list[str] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                text_parts.append(page_text.strip())

    full_text = "\n\n".join(text_parts)
    logger.info("Extracted %d characters from %d pages", len(full_text), len(text_parts))
    return full_text


# ──────────────────────────────────────────────
# POST /ingest/url
# ──────────────────────────────────────────────

@router.post("/url", response_model=IngestResponse)
async def ingest_url(request: IngestURLRequest) -> IngestResponse:
    """
    Ingest a web page by URL.

    Scrapes the page using Firecrawl, extracts clean text,
    chunks it, generates embeddings, and stores in the vector database.
    """
    logger.info("Ingesting URL: %s for user %s", request.url, request.user_id)

    # Scrape the URL
    try:
        scraped = await scrape_url(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("URL scraping failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scrape URL: {exc}",
        )

    # Use provided title or scraped title
    url_title = request.title or scraped["title"]

    return await _ingest_pipeline(
        user_id=request.user_id,
        title=url_title,
        source_type=SourceType.URL,
        raw_text=scraped["content"],
        source_url=request.url,
    )


# ──────────────────────────────────────────────
# POST /ingest/voice
# ──────────────────────────────────────────────

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac"}

@router.post("/voice", response_model=IngestResponse)
async def ingest_voice(
    file: UploadFile = File(..., description="Audio file to transcribe and ingest"),
    title: str = Form(None, description="Optional title (defaults to 'Voice Note — <timestamp>')"),
    user_id: str = Form(..., description="UUID of the authenticated user"),
) -> IngestResponse:
    """
    Ingest a voice recording as a knowledge item.

    Transcribes audio using Groq Whisper Large V3, then chunks the
    transcript, generates embeddings, and stores in the vector database.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{file_ext}'. Supported: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file is too large. Maximum size is {MAX_AUDIO_SIZE // (1024 * 1024)} MB.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    # Generate default title if not provided
    from datetime import datetime, timezone
    voice_title = title or f"Voice Note — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"

    logger.info(
        "Ingesting voice: '%s' (%d bytes, %s) for user %s",
        voice_title,
        len(content),
        file_ext,
        user_id,
    )

    # Save to temporary file for Groq Whisper
    try:
        with tempfile.NamedTemporaryFile(
            suffix=file_ext,
            delete=False,
            dir=tempfile.gettempdir(),
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Transcribe with Groq Whisper
        transcript = await transcribe_audio(tmp_path)

        if not transcript or not transcript.strip():
            raise HTTPException(
                status_code=400,
                detail="No speech could be detected in this audio file.",
            )

        logger.info("Transcribed %d characters from audio", len(transcript))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Voice transcription failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transcribe audio: {exc}",
        )
    finally:
        # Clean up the temp file
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass

    return await _ingest_pipeline(
        user_id=user_id,
        title=voice_title,
        source_type=SourceType.VOICE,
        raw_text=transcript,
    )
