"""
AI Second Brain — FastAPI Backend Entry Point.
Configures CORS, registers all routers, loads environment variables,
and exposes a health check endpoint for monitoring service status.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import HealthResponse
from services.embedder import check_ollama_health
from services.vector_store import check_supabase_health
from services.groq_client import check_groq_health

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("🧠 AI Second Brain — Starting up...")
    logger.info("=" * 60)

    # Check service health on startup
    ollama_ok = await check_ollama_health()
    supabase_ok = await check_supabase_health()

    if ollama_ok:
        logger.info("✅ Ollama (nomic-embed-text) — Connected")
    else:
        logger.warning("⚠️  Ollama — Not available (embeddings will fail)")

    if supabase_ok:
        logger.info("✅ Supabase (pgvector) — Connected")
    else:
        logger.warning("⚠️  Supabase — Not available (storage will fail)")

    logger.info("🚀 Backend ready at http://localhost:8000")
    logger.info("📚 API docs at http://localhost:8000/docs")

    yield  # App is running

    logger.info("🛑 AI Second Brain — Shutting down...")


# Create the FastAPI application
app = FastAPI(
    title="AI Second Brain API",
    description=(
        "Personal knowledge base API with RAG-powered chat. "
        "Upload PDFs, URLs, voice notes, and text — then chat with all of it."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Check the health of all connected services.
    Returns status for Ollama, Supabase, and Groq.
    """
    ollama_status = "healthy" if await check_ollama_health() else "unavailable"
    supabase_status = "healthy" if await check_supabase_health() else "unavailable"
    groq_status = "healthy" if await check_groq_health() else "unavailable"

    overall = "healthy" if all(
        s == "healthy" for s in [ollama_status, supabase_status, groq_status]
    ) else "degraded"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        services={
            "ollama": ollama_status,
            "supabase": supabase_status,
            "groq": groq_status,
        },
    )


@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    """Root endpoint — confirms the API is running."""
    return {
        "message": "🧠 AI Second Brain API is running",
        "docs": "/docs",
        "health": "/health",
    }


# ──────────────────────────────────────────────
# Register Routers (will be added in Phase 2 & 3)
# ──────────────────────────────────────────────
# Uncomment these as the routers are built:
# from routers.ingest import router as ingest_router
# from routers.chat import router as chat_router
# from routers.knowledge import router as knowledge_router
#
# app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
# app.include_router(chat_router, prefix="/chat", tags=["Chat"])
# app.include_router(knowledge_router, prefix="/knowledge", tags=["Knowledge"])
