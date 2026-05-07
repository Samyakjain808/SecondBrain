"""
Pydantic models for all API request/response schemas.
Provides strict typing and validation for every endpoint in the Second Brain API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class SourceType(str, Enum):
    """Allowed knowledge source types."""
    PDF = "pdf"
    URL = "url"
    VOICE = "voice"
    TEXT = "text"


# ──────────────────────────────────────────────
# Ingestion Requests
# ──────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    """Request body for ingesting raw text."""
    title: str = Field(..., min_length=1, max_length=500, description="Title for the knowledge item")
    content: str = Field(..., min_length=1, description="Raw text content to ingest")
    user_id: str = Field(..., description="UUID of the authenticated user")


class IngestURLRequest(BaseModel):
    """Request body for ingesting a web page by URL."""
    url: str = Field(..., description="URL to scrape and ingest")
    title: Optional[str] = Field(None, max_length=500, description="Optional title override")
    user_id: str = Field(..., description="UUID of the authenticated user")


# Note: PDF and Voice ingestion use multipart form-data (file uploads),
# so their schemas are defined inline in the router with Form() + File().


# ──────────────────────────────────────────────
# Ingestion Responses
# ──────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response after successfully ingesting content."""
    knowledge_item_id: str = Field(..., description="UUID of the created knowledge item")
    title: str = Field(..., description="Title of the ingested item")
    source_type: SourceType = Field(..., description="Type of the source")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    tags: list[str] = Field(default_factory=list, description="Auto-generated tags")
    message: str = Field(default="Ingestion successful", description="Status message")


# ──────────────────────────────────────────────
# Chat Models
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the RAG chat endpoint."""
    query: str = Field(..., min_length=1, max_length=5000, description="User's question")
    user_id: str = Field(..., description="UUID of the authenticated user")
    use_reranking: bool = Field(default=True, description="Whether to use Cohere re-ranking")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of chunks to retrieve")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of chunks after re-ranking")


class SourceChunk(BaseModel):
    """A source chunk returned alongside a chat response."""
    chunk_id: str = Field(..., description="UUID of the chunk")
    content: str = Field(..., description="Text content of the chunk")
    knowledge_item_id: str = Field(..., description="UUID of the parent knowledge item")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    title: Optional[str] = Field(None, description="Title of the parent knowledge item")
    source_type: Optional[str] = Field(None, description="Source type of the parent item")


class ChatStreamEvent(BaseModel):
    """A single event in the chat stream (SSE payload)."""
    event: str = Field(..., description="Event type: 'token', 'sources', 'done', 'error'")
    data: str = Field(default="", description="Token text or JSON payload")


# ──────────────────────────────────────────────
# Knowledge Item Models
# ──────────────────────────────────────────────

class KnowledgeItem(BaseModel):
    """A knowledge item in the database."""
    id: str = Field(..., description="UUID of the knowledge item")
    user_id: str = Field(..., description="UUID of the owner")
    title: str = Field(..., description="Title of the item")
    source_type: SourceType = Field(..., description="Source type")
    source_url: Optional[str] = Field(None, description="Source URL if applicable")
    raw_text: Optional[str] = Field(None, description="Full raw text of the item")
    created_at: Optional[str] = Field(None, description="ISO timestamp of creation")
    tags: list[str] = Field(default_factory=list, description="Tags for the item")
    chunk_count: Optional[int] = Field(None, description="Number of associated chunks")


class KnowledgeListResponse(BaseModel):
    """Response for listing knowledge items."""
    items: list[KnowledgeItem] = Field(default_factory=list, description="List of knowledge items")
    total: int = Field(default=0, ge=0, description="Total count of items")


class KnowledgeDeleteResponse(BaseModel):
    """Response after deleting a knowledge item."""
    deleted_id: str = Field(..., description="UUID of the deleted item")
    message: str = Field(default="Knowledge item deleted successfully")


class KnowledgeSearchRequest(BaseModel):
    """Request body for semantic search across knowledge items."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    user_id: str = Field(..., description="UUID of the authenticated user")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")


class KnowledgeSearchResult(BaseModel):
    """A single search result with similarity score."""
    chunk_id: str = Field(..., description="UUID of the matching chunk")
    content: str = Field(..., description="Text content of the matching chunk")
    knowledge_item_id: str = Field(..., description="UUID of the parent knowledge item")
    title: Optional[str] = Field(None, description="Title of the parent knowledge item")
    source_type: Optional[str] = Field(None, description="Source type of the parent item")
    similarity: float = Field(..., description="Cosine similarity score")


class KnowledgeSearchResponse(BaseModel):
    """Response for semantic search."""
    results: list[KnowledgeSearchResult] = Field(default_factory=list, description="Search results")
    query: str = Field(..., description="Original search query")


# ──────────────────────────────────────────────
# Dashboard Stats
# ──────────────────────────────────────────────

class DashboardStats(BaseModel):
    """Aggregated statistics for the user dashboard."""
    total_items: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    sources_by_type: dict[str, int] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    services: dict[str, str] = Field(default_factory=dict)
