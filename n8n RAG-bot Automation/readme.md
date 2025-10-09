# RAG Chatbot System for Bitovi Blog Articles

A Retrieval-Augmented Generation (RAG) chatbot built with n8n, PostgreSQL, Qdrant, and OpenAI that intelligently answers questions about Bitovi's blog content through automated scraping, semantic search, and metadata filtering.

---

##  Overview

This system automatically scrapes, indexes, and enables semantic search across Bitovi's blog articles through three integrated workflows:

1. **Web Scraping Pipeline** - Automated article discovery and PostgreSQL storage
2. **Vector Embedding Pipeline** - Semantic indexing with OpenAI embeddings in Qdrant
3. **Dual-Mode Chatbot** - Intelligent query routing for content search and metadata filtering

---

##  Key Features

-  **Automated Knowledge Base Updates** - Scheduled scraping keeps content current
-  **Multi-Layer Duplicate Prevention** - Checks at both PostgreSQL and Qdrant levels
-  **Intelligent Query Classification** - Routes questions to semantic search or metadata filtering
-  **Source Attribution** - All answers include clickable article citations
-  **Hallucination Prevention** - Strict context adherence ensures accuracy
-  **Graceful Error Handling** - Clean responses when no relevant articles are found

---

##  Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | n8n |
| Relational DB | PostgreSQL |
| Vector DB | Qdrant |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Generation | OpenAI GPT-4o-Mini |
| Languages | Python (scraping), JavaScript (transforms) |

---

##  Prerequisites

- n8n instance (self-hosted or cloud)
- PostgreSQL database
- Qdrant vector database
- OpenAI API key

---

##  Workflows

### Workflow 1: Web Scraping & Storage

**Purpose:** Scrapes Bitovi blog articles and stores metadata in PostgreSQL

**Features:**
- Incremental page-by-page insertion
- Automatic pagination handling (41 pages)
- Duplicate prevention via `ON CONFLICT (url) DO NOTHING`
- Processes 150+ articles

---

### Workflow 2: Vector Embedding Pipeline

**Purpose:** Generates embeddings for articles and stores them in Qdrant

**Features:**
- Pre-embedding duplicate detection
- Batch processing capability
- Structured metadata for rich retrieval

---

### Workflow 3: RAG Chatbot

**Purpose:** Answers user questions through semantic search or metadata filtering

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Similarity Threshold | 0.3 | Minimum relevance for results |
| Result Limit | 5 | Max articles returned per query |
| Embedding Model | text-embedding-3-small | 1536-dim vectors |
| Generation Model | GPT-4o-Mini | Fast, accurate responses |
| Scraping Schedule | Daily at midnight | Off-peak processing |

---

## Getting Started

### 1. Setup Databases

**PostgreSQL:**
```sql
CREATE TABLE bitovi_articles_test (
    id INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    author TEXT,
    topic TEXT,
    summary TEXT
);
```

**Qdrant:**
- Create collection: `bitovi_articles`
- Vector size: 1536
- Distance metric: Cosine

### 2. Configure n8n

**Import workflows and set credentials:**
- PostgreSQL connection
- Qdrant API key
- OpenAI API key (Header Auth)

### 3. Execute Workflows

1. **Run Scraping Workflow** → Populates PostgreSQL with articles
2. **Run Embedding Workflow** → Creates vectors in Qdrant
3. **Activate Chatbot** → Start asking questions in chat interface!

---

## Technical Details

### Duplicate Prevention Strategy

**Three-Layer Approach:**
1. **Scraping:** PostgreSQL UNIQUE constraint blocks duplicate URLs
2. **Embedding:** Qdrant check before calling OpenAI API
3. **Upsert:** `ON CONFLICT DO NOTHING` for safe workflow re-runs

### Query Classification Logic

AI classifier returns:
- **"metadata"** → Routes to topic filtering (counting/listing)
- **"content"** → Routes to vector similarity search (RAG)

