-- =============================================================
-- AI Second Brain — Supabase Database Schema
-- =============================================================
-- Run this SQL in your Supabase SQL Editor:
-- https://supabase.com → Project → SQL Editor → New Query
-- =============================================================

-- Enable pgvector extension for vector similarity search
create extension if not exists vector;

-- ──────────────────────────────────────────────
-- Knowledge Items Table (each uploaded source)
-- ──────────────────────────────────────────────
create table knowledge_items (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  title text not null,
  source_type text not null check (source_type in ('pdf', 'url', 'voice', 'text')),
  source_url text,
  raw_text text,
  created_at timestamptz default now(),
  tags text[] default '{}'
);

-- ──────────────────────────────────────────────
-- Chunks Table (each embedded piece of text)
-- ──────────────────────────────────────────────
create table chunks (
  id uuid primary key default gen_random_uuid(),
  knowledge_item_id uuid references knowledge_items(id) on delete cascade,
  user_id uuid references auth.users(id) on delete cascade,
  content text not null,
  embedding vector(768),           -- nomic-embed-text dimension
  chunk_index integer,
  token_count integer,
  created_at timestamptz default now()
);

-- ──────────────────────────────────────────────
-- Index for fast cosine similarity search
-- ──────────────────────────────────────────────
create index on chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- ──────────────────────────────────────────────
-- Row Level Security (RLS)
-- ──────────────────────────────────────────────
alter table knowledge_items enable row level security;
alter table chunks enable row level security;

create policy "Users can only access own items"
  on knowledge_items for all using (auth.uid() = user_id);

create policy "Users can only access own chunks"
  on chunks for all using (auth.uid() = user_id);

-- ──────────────────────────────────────────────
-- Similarity Search Function (RPC)
-- ──────────────────────────────────────────────
create or replace function match_chunks(
  query_embedding vector(768),
  match_user_id uuid,
  match_count int default 10
)
returns table (
  id uuid,
  content text,
  knowledge_item_id uuid,
  similarity float
)
language sql stable
as $$
  select
    chunks.id,
    chunks.content,
    chunks.knowledge_item_id,
    1 - (chunks.embedding <=> query_embedding) as similarity
  from chunks
  where chunks.user_id = match_user_id
  order by chunks.embedding <=> query_embedding
  limit match_count;
$$;
