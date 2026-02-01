# Q&A RAG Bot

A RAG (Retrieval Augmented Generation) system that crawls websites and answers questions based on the crawled content.

## What it does

1. Crawls a website and grabs the text
2. Chunks the text and generates embeddings (OpenAI)
3. Stores everything in ChromaDB
4. When you ask a question, it finds relevant chunks and uses GPT to answer

## Setup

```bash
# create venv
python -m venv venv
source venv/bin/activate

# install deps
pip install -r requirements.txt

# set your openai key
cp .env.example .env
# edit .env and add OPENAI_API_KEY
```

## Usage

### Crawl a site

```bash
python main.py crawl https://docs.example.com --max-pages 50
```

### Ask questions

```bash
python main.py ask "How do I install it?"
python main.py ask "What features are available?" --show-sources
```

### Start the API

```bash
python main.py serve
# docs at http://localhost:8000/docs
```

### Interactive mode

```bash
python main.py interactive
```

## API endpoints

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/api/ask` | POST | Ask a question |
| `/api/chat` | POST | Chat with history |
| `/api/crawl` | POST | Crawl a website |
| `/api/search` | GET | Search without generating answer |
| `/api/stats` | GET | Get collection stats |
| `/api/clear` | DELETE | Clear all data |
| `/health` | GET | Health check |

### Example curl

```bash
# ask a question
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'

# crawl a site
curl -X POST "http://localhost:8000/api/crawl" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.example.com", "max_pages": 50}'
```

## Config

Set these in `.env`:

- `OPENAI_API_KEY` - required
- `EMBEDDING_MODEL` - default: text-embedding-3-small
- `LLM_MODEL` - default: gpt-3.5-turbo
- `CHUNK_SIZE` - default: 1000
- `CHUNK_OVERLAP` - default: 200
- `TOP_K_RESULTS` - default: 5
- `MAX_PAGES` - default: 100

## Project structure

```
├── src/
│   ├── config.py         # settings
│   ├── crawler.py        # web crawler
│   ├── text_processor.py # chunking
│   ├── vector_store.py   # chromadb + embeddings
│   ├── rag_engine.py     # retrieval + generation
│   └── api.py            # fastapi app
├── tests/
├── main.py               # cli
├── requirements.txt
└── .env.example
```

## Running tests

```bash
pytest tests/ -v
```

## Troubleshooting

**"OpenAI API key required"** - set OPENAI_API_KEY in .env

**"No docs in store"** - run crawl first

**Low quality answers** - try increasing n_results or crawl more pages
