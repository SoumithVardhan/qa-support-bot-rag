import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .crawler import WebCrawler
from .text_processor import process_documents
from .vector_store import VectorStore
from .rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_store = None
rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, rag_engine
    logger.info("Starting up...")
    try:
        vector_store = VectorStore()
        rag_engine = RAGEngine(vector_store=vector_store)
        logger.info("RAG engine ready")
    except Exception as e:
        logger.error(f"Init failed: {e}")
        raise
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Q&A RAG Bot",
    description="RAG-based Q&A API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# request/response models

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    n_results: Optional[int] = Field(default=5, ge=1, le=20)
    temperature: Optional[float] = Field(default=0.3, ge=0, le=1)


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    chat_history: Optional[List[ChatMessage]] = Field(default=[])
    n_results: Optional[int] = Field(default=5, ge=1, le=20)


class SourceInfo(BaseModel):
    url: str
    title: str
    relevance_score: float
    snippet: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    query: str
    confidence: float


class CrawlRequest(BaseModel):
    url: str = Field(...)
    max_pages: Optional[int] = Field(default=50, ge=1, le=500)
    clear_existing: Optional[bool] = Field(default=False)


class CrawlResponse(BaseModel):
    status: str
    message: str
    pages_crawled: Optional[int] = None
    chunks_created: Optional[int] = None


class StatsResponse(BaseModel):
    collection_name: str
    document_count: int
    persist_directory: str


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    rag_engine_ready: bool
    document_count: int


# endpoints

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Q&A RAG Bot",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/api/ask",
            "chat": "/api/chat",
            "crawl": "/api/crawl",
            "stats": "/api/stats",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    global vector_store, rag_engine
    doc_count = 0
    vs_ok = vector_store is not None
    rag_ok = rag_engine is not None

    if vs_ok:
        try:
            stats = vector_store.get_collection_stats()
            doc_count = stats.get("document_count", 0)
        except:
            vs_ok = False

    return HealthResponse(
        status="healthy" if (vs_ok and rag_ok) else "degraded",
        vector_store_ready=vs_ok,
        rag_engine_ready=rag_ok,
        document_count=doc_count
    )


@app.post("/api/ask", response_model=AnswerResponse, tags=["Q&A"])
async def ask(request: QuestionRequest):
    global rag_engine
    if not rag_engine:
        raise HTTPException(503, "RAG engine not ready")

    try:
        resp = rag_engine.query(
            question=request.question,
            n_results=request.n_results,
            temperature=request.temperature
        )
        return AnswerResponse(
            answer=resp.answer,
            sources=[SourceInfo(url=s["url"], title=s["title"],
                               relevance_score=s["relevance_score"], snippet=s["snippet"])
                     for s in resp.sources],
            query=resp.query,
            confidence=resp.confidence
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/chat", response_model=AnswerResponse, tags=["Q&A"])
async def chat(request: ChatRequest):
    global rag_engine
    if not rag_engine:
        raise HTTPException(503, "RAG engine not ready")

    try:
        history = [{"role": m.role, "content": m.content} for m in request.chat_history]
        resp = rag_engine.query_with_history(
            question=request.question,
            chat_history=history,
            n_results=request.n_results
        )
        return AnswerResponse(
            answer=resp.answer,
            sources=[SourceInfo(url=s["url"], title=s["title"],
                               relevance_score=s["relevance_score"], snippet=s["snippet"])
                     for s in resp.sources],
            query=resp.query,
            confidence=resp.confidence
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/crawl", response_model=CrawlResponse, tags=["Data"])
async def crawl(request: CrawlRequest):
    global vector_store
    if not vector_store:
        raise HTTPException(503, "Vector store not ready")

    try:
        logger.info(f"Crawling {request.url}")

        if request.clear_existing:
            vector_store.clear_collection()

        crawler = WebCrawler(base_url=request.url, max_pages=request.max_pages)
        crawler.crawl()
        docs = crawler.get_documents()

        if not docs:
            return CrawlResponse(status="warning", message="No pages crawled",
                                pages_crawled=0, chunks_created=0)

        chunks = process_documents(docs)
        num_added = vector_store.add_documents(chunks)

        return CrawlResponse(
            status="success",
            message=f"Crawled {len(docs)} pages, created {num_added} chunks",
            pages_crawled=len(docs),
            chunks_created=num_added
        )
    except Exception as e:
        logger.error(f"Crawl error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/stats", response_model=StatsResponse, tags=["Data"])
async def stats():
    global vector_store
    if not vector_store:
        raise HTTPException(503, "Vector store not ready")
    try:
        s = vector_store.get_collection_stats()
        return StatsResponse(**s)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(500, str(e))


@app.delete("/api/clear", tags=["Data"])
async def clear():
    global vector_store
    if not vector_store:
        raise HTTPException(503, "Vector store not ready")
    try:
        vector_store.clear_collection()
        return {"status": "success", "message": "Data cleared"}
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/search", tags=["Q&A"])
async def search(
    query: str = Query(..., min_length=1),
    n_results: int = Query(default=5, ge=1, le=20)
):
    global vector_store
    if not vector_store:
        raise HTTPException(503, "Vector store not ready")

    try:
        results = vector_store.search(query, n_results=n_results)
        return {
            "query": query,
            "results": [{"content": r["content"], "metadata": r["metadata"],
                        "relevance_score": r["relevance_score"]} for r in results]
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
