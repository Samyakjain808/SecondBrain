# 🧠 SecondBrain

> AI-powered personal knowledge base — dump PDFs, URLs, voice notes, and text, then chat with all of it using RAG.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (App Router) + Tailwind CSS + shadcn/ui |
| Backend | FastAPI (Python 3.11+) |
| LLM | Groq API — LLaMA 3.3 70B |
| Voice STT | Groq Whisper Large V3 |
| Embeddings | nomic-embed-text via Ollama (free, local) |
| Vector DB | Supabase with pgvector |
| Re-ranking | Cohere Rerank API |

## Project Structure

```
second-brain/
├── backend/             ← FastAPI Python backend
│   ├── main.py          ← App entry point, CORS, health checks
│   ├── models/          ← Pydantic schemas
│   ├── routers/         ← API route handlers
│   └── services/        ← Core business logic
├── frontend/            ← Next.js 14 frontend (coming soon)
└── database/            ← Supabase SQL schema
```

## Setup

1. Clone the repo
2. Set up API keys in `backend/.env` (see `.env` template)
3. Install Ollama & pull embedding model: `ollama pull nomic-embed-text`
4. Install Python deps: `pip install -r backend/requirements.txt`
5. Run the backend: `uvicorn main:app --reload --port 8000`

## Status

- ✅ Phase 1 — Backend Foundation (complete)
- ⬜ Phase 2 — Ingestion Routes
- ⬜ Phase 3 — RAG Chat Route
- ⬜ Phase 4–8 — Frontend & Deployment
