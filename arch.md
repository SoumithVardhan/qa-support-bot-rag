# Q&A RAG Bot - Architecture Documentation

## Overview

This project implements a production-ready Retrieval-Augmented Generation (RAG) system that enables question-answering over crawled website content. The system crawls websites, processes and chunks the content, stores embeddings in a vector database, and uses an LLM to generate contextual answers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Q&A RAG Bot                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐  │
│   │  Web     │───▶│    Text      │───▶│   Vector    │───▶│     RAG      │  │
│   │ Crawler  │    │  Processor   │    │    Store    │    │   Engine     │  │
│   └──────────┘    └──────────────┘    └─────────────┘    └──────────────┘  │
│        │                                      │                  │          │
│        │                                      │                  ▼          │
│        ▼                                      │          ┌──────────────┐  │
│   ┌──────────┐                               │          │   FastAPI    │  │
│   │ robots.  │                               │          │     API      │  │
│   │   txt    │                               ▼          └──────────────┘  │
│   └──────────┘                        ┌─────────────┐           │          │
│                                       │  ChromaDB   │           ▼          │
│                                       │  (Persist)  │    ┌──────────────┐  │
│                                       └─────────────┘    │    CLI       │  │
│                                                          │  Interface   │  │
│                                                          └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## System Components

### 1. Web Crawler (`src/crawler.py`)

The crawler is responsible for fetching and extracting content from websites. It implements production-ready features for reliable and polite crawling.

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `WebCrawler` | Main crawler orchestrating the crawl process |
| `RobotsChecker` | Handles robots.txt parsing and compliance |
| `DomainRateLimiter` | Implements per-domain rate limiting |
| `RetryConfig` | Configuration for exponential backoff retry |
| `CrawledPage` | Data class for crawled page content |

**Features:**

- **Robots.txt Compliance**: Respects `robots.txt` rules and `Crawl-delay` directives
- **Exponential Backoff**: Retries failed requests with configurable backoff strategy
- **Per-Domain Rate Limiting**: Prevents overwhelming individual servers
- **Jitter**: Adds randomness to retry delays to prevent thundering herd
- **Content Extraction**: Intelligently extracts main content, removing boilerplate

**Retry Logic Flow:**

```
Request Failed?
     │
     ▼
┌────────────────┐     No      ┌─────────────┐
│ Retryable?     │────────────▶│   Fail      │
│ (timeout/5xx)  │             └─────────────┘
└────────────────┘
     │ Yes
     ▼
┌────────────────┐     No      ┌─────────────┐
│ Retries left?  │────────────▶│   Fail      │
└────────────────┘             └─────────────┘
     │ Yes
     ▼
┌────────────────┐
│ Calculate delay│
│ with backoff   │
│ + jitter       │
└────────────────┘
     │
     ▼
┌────────────────┐
│ Wait & Retry   │
└────────────────┘
```

---

### 2. Text Processor (`src/text_processor.py`)

Handles cleaning and chunking of crawled content for optimal embedding and retrieval.

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `TextCleaner` | Removes boilerplate, normalizes whitespace |
| `TextChunker` | Splits documents into semantic chunks |
| `TextChunk` | Data class for processed chunks |

**Processing Pipeline:**

```
Raw Content
     │
     ▼
┌─────────────────┐
│  Clean Text     │  ← Remove special chars, normalize whitespace
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Remove         │  ← Strip copyright, privacy policy, etc.
│  Boilerplate    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Recursive      │  ← Split by paragraphs, sentences, words
│  Chunking       │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Token Count    │  ← Calculate tokens per chunk
└─────────────────┘
     │
     ▼
Processed Chunks
```

**Configuration:**

- `chunk_size`: Target size for each chunk (default: 1000 chars)
- `chunk_overlap`: Overlap between chunks for context continuity (default: 200 chars)

---

### 3. Vector Store (`src/vector_store.py`)

Manages embeddings storage and similarity search using ChromaDB.

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `EmbeddingGenerator` | Creates embeddings via OpenAI API |
| `VectorStore` | ChromaDB wrapper for storage and search |

**Architecture:**

```
┌─────────────────────────────────────────────────────┐
│                    VectorStore                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────┐    ┌───────────────────────┐ │
│  │ EmbeddingGenerator│───▶│  OpenAI Embeddings   │ │
│  │                   │    │  (text-embedding-3)  │ │
│  └──────────────────┘    └───────────────────────┘ │
│           │                                         │
│           ▼                                         │
│  ┌──────────────────────────────────────────────┐  │
│  │                  ChromaDB                     │  │
│  │  ┌─────────────────────────────────────────┐ │  │
│  │  │  Collection: website_docs               │ │  │
│  │  │  - documents (text)                     │ │  │
│  │  │  - embeddings (vectors)                 │ │  │
│  │  │  - metadata (url, title, chunk_id)      │ │  │
│  │  └─────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Operations:**

- `add_documents()`: Batch insert chunks with embeddings
- `search()`: Cosine similarity search for relevant chunks
- `delete_by_url()`: Remove chunks from a specific source
- `clear_collection()`: Reset the entire collection

---

### 4. RAG Engine (`src/rag_engine.py`)

Orchestrates the retrieval and generation pipeline for answering questions.

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `RAGEngine` | Main engine coordinating retrieval and generation |
| `RAGResponse` | Structured response with answer, sources, confidence |

**Query Pipeline:**

```
User Question
     │
     ▼
┌─────────────────┐
│   Retrieve      │  ← Vector similarity search
│   (top-k docs)  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Format Context  │  ← Combine sources with metadata
└─────────────────┘
     │
     ▼
┌─────────────────┐
│   Generate      │  ← LLM generates answer from context
│   Answer        │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│   Calculate     │  ← Based on retrieval scores
│   Confidence    │
└─────────────────┘
     │
     ▼
RAGResponse
```

**Features:**

- Context-aware answer generation
- Source attribution
- Confidence scoring
- Chat history support for follow-up questions

---

### 5. API Server (`src/api.py`)

RESTful API built with FastAPI for external integrations.

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and available endpoints |
| GET | `/health` | Health check with component status |
| POST | `/api/ask` | Single question Q&A |
| POST | `/api/chat` | Multi-turn conversation |
| POST | `/api/crawl` | Trigger website crawl |
| GET | `/api/stats` | Vector store statistics |
| DELETE | `/api/clear` | Clear all stored data |
| GET | `/api/search` | Direct vector search |

**Request/Response Flow:**

```
Client Request
     │
     ▼
┌─────────────────┐
│   FastAPI       │
│   Validation    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│   RAG Engine    │
│   Processing    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│   Pydantic      │
│   Response      │
└─────────────────┘
     │
     ▼
JSON Response
```

---

### 6. CLI Interface (`main.py`)

Command-line interface for direct interaction.

**Commands:**

| Command | Description |
|---------|-------------|
| `crawl <url>` | Crawl a website and index content |
| `ask <question>` | Ask a single question |
| `interactive` | Start interactive Q&A session |
| `serve` | Start the API server |
| `stats` | Show vector store statistics |
| `clear` | Clear all indexed data |

---

## Data Flow

### Indexing Flow (Crawl → Store)

```
┌─────────┐   URLs    ┌─────────┐  Pages   ┌─────────┐  Chunks  ┌─────────┐
│ Website │─────────▶│ Crawler │─────────▶│Processor│─────────▶│  Store  │
└─────────┘          └─────────┘          └─────────┘          └─────────┘
                          │                                          │
                          ▼                                          ▼
                    ┌───────────┐                            ┌───────────┐
                    │robots.txt │                            │ ChromaDB  │
                    │  check    │                            │(Persisted)│
                    └───────────┘                            └───────────┘
```

### Query Flow (Question → Answer)

```
┌──────────┐  Query   ┌─────────┐  Embed   ┌─────────┐  Docs   ┌─────────┐
│   User   │─────────▶│   RAG   │─────────▶│  Store  │────────▶│   RAG   │
└──────────┘          │ Engine  │          │ Search  │         │ Engine  │
                      └─────────┘          └─────────┘         └─────────┘
                                                                    │
     ┌──────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────┐  Prompt  ┌─────────┐  Answer  ┌──────────┐
│ Context │─────────▶│   LLM   │─────────▶│   User   │
│ Format  │          │ (GPT)   │          │          │
└─────────┘          └─────────┘          └──────────┘
```

---

## Configuration

All settings are managed through environment variables with sensible defaults.

### Environment Variables

```bash
# Core
OPENAI_API_KEY=sk-...           # Required

# Models
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo

# Crawler - Basic
MAX_PAGES=100
CRAWL_DELAY=1.0

# Crawler - Retry (Exponential Backoff)
MAX_RETRIES=3
RETRY_BASE_DELAY=1.0
RETRY_MAX_DELAY=60.0
RETRY_EXPONENTIAL_BASE=2.0
RETRY_JITTER=true

# Crawler - Politeness
RESPECT_ROBOTS_TXT=true
ROBOTS_CACHE_TTL=3600

# Storage
CHROMA_PERSIST_DIR=./data/chroma_db
COLLECTION_NAME=website_docs

# Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

---

## Key Design Decisions

### 1. Robots.txt Compliance
The crawler respects website owner preferences by checking robots.txt before crawling. This includes honoring `Crawl-delay` directives and disallowed paths.

### 2. Exponential Backoff with Jitter
Failed requests are retried with exponential backoff to handle transient failures gracefully. Jitter prevents synchronized retry storms.

### 3. Per-Domain Rate Limiting
The crawler maintains separate rate limits for each domain, ensuring polite crawling behavior even when crawling multiple sites.

### 4. Chunking Strategy
Recursive text splitting preserves semantic boundaries (paragraphs → sentences → words) while maintaining configurable chunk sizes and overlap for context continuity.

### 5. ChromaDB for Persistence
ChromaDB provides efficient vector storage with persistence, enabling the system to maintain its knowledge base across restarts.

### 6. Confidence Scoring
Answers include confidence scores based on retrieval relevance, helping users gauge answer reliability.

---

## Error Handling

| Component | Error Type | Handling Strategy |
|-----------|------------|-------------------|
| Crawler | Network timeout | Retry with exponential backoff |
| Crawler | HTTP 429/5xx | Retry with Retry-After header respect |
| Crawler | robots.txt blocked | Skip URL, log warning |
| Embeddings | API error | Propagate with context |
| Vector Store | Query failure | Return empty results |
| RAG Engine | No relevant docs | Return "insufficient info" response |

---

## Performance Considerations

- **Batch Processing**: Embeddings are generated in batches to minimize API calls
- **Connection Pooling**: HTTP session reuse for crawler efficiency
- **Caching**: robots.txt files are cached to reduce redundant fetches
- **Async Support**: API endpoints support concurrent requests
- **Persistent Storage**: ChromaDB persists to disk, avoiding re-indexing

---

## Security Notes

- API keys are loaded from environment variables (never hardcoded)
- CORS is configured (customize for production)
- Input validation via Pydantic models
- Respectful crawling prevents abuse detection/blocking

---

## Future Enhancements

- [ ] Async crawler for improved throughput
- [ ] Sitemap.xml parsing for URL discovery
- [ ] Multi-collection support for topic separation
- [ ] Hybrid search (keyword + semantic)
- [ ] Answer caching for common questions
- [ ] Webhook notifications for crawl completion
- [ ] Authentication for API endpoints
