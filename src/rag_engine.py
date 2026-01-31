import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from openai import OpenAI

from .config import settings
from .vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    confidence: float


class RAGEngine:
    SYSTEM_PROMPT = """You are a helpful Q&A assistant. Answer questions based ONLY on the provided context.

Rules:
- Only use information from the context below
- If the context doesn't have the answer, say you don't have enough info
- Be concise but complete
- Mention which source has the info when relevant
- Don't make stuff up

Context format:
[Source: URL]
Content...
"""

    def __init__(self, vector_store=None, model=None, api_key=None, top_k=None):
        self.vector_store = vector_store or VectorStore()
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.openai_api_key
        self.top_k = top_k or settings.top_k_results

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        self.client = OpenAI(api_key=self.api_key)

    def retrieve(self, query, n_results=None):
        n = n_results or self.top_k
        results = self.vector_store.search(query, n_results=n)
        logger.info(f"Retrieved {len(results)} docs for: {query[:50]}...")
        return results

    def format_context(self, docs):
        parts = []
        for i, doc in enumerate(docs, 1):
            url = doc.get("metadata", {}).get("url", "Unknown")
            title = doc.get("metadata", {}).get("title", "Untitled")
            content = doc.get("content", "")
            parts.append(f"[Source {i}: {title}]\nURL: {url}\nContent: {content}\n")
        return "\n---\n".join(parts)

    def generate_answer(self, query, context, temperature=0.3):
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        return resp.choices[0].message.content

    def calc_confidence(self, docs):
        if not docs:
            return 0.0
        scores = [d.get("relevance_score", 0) for d in docs[:3]]
        if not scores:
            return 0.0
        avg = sum(scores) / len(scores)
        return max(0.0, min(1.0, avg))

    def query(self, question, n_results=None, temperature=0.3):
        logger.info(f"Processing: {question}")

        docs = self.retrieve(question, n_results)
        if not docs:
            return RAGResponse(
                answer="I couldn't find relevant info to answer your question.",
                sources=[], query=question, confidence=0.0
            )

        context = self.format_context(docs)
        answer = self.generate_answer(question, context, temperature)
        confidence = self.calc_confidence(docs)

        sources = [{
            "url": d.get("metadata", {}).get("url", ""),
            "title": d.get("metadata", {}).get("title", ""),
            "relevance_score": d.get("relevance_score", 0),
            "snippet": d.get("content", "")[:200] + "..."
        } for d in docs]

        return RAGResponse(answer=answer, sources=sources, query=question, confidence=confidence)

    def query_with_history(self, question, chat_history=None, n_results=None):
        chat_history = chat_history or []

        docs = self.retrieve(question, n_results)
        if not docs:
            return RAGResponse(
                answer="I couldn't find relevant info to answer your question.",
                sources=[], query=question, confidence=0.0
            )

        context = self.format_context(docs)

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        # add last few messages from history
        for msg in chat_history[-10:]:
            messages.append(msg)
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.3, max_tokens=1000
        )

        answer = resp.choices[0].message.content
        confidence = self.calc_confidence(docs)

        sources = [{
            "url": d.get("metadata", {}).get("url", ""),
            "title": d.get("metadata", {}).get("title", ""),
            "relevance_score": d.get("relevance_score", 0),
            "snippet": d.get("content", "")[:200] + "..."
        } for d in docs]

        return RAGResponse(answer=answer, sources=sources, query=question, confidence=confidence)


def create_rag_engine(collection_name=None):
    vs = VectorStore(collection_name=collection_name)
    return RAGEngine(vector_store=vs)


if __name__ == "__main__":
    engine = create_rag_engine()
    stats = engine.vector_store.get_collection_stats()
    print(f"Collection: {stats}")

    if stats["document_count"] > 0:
        q = "What is this documentation about?"
        resp = engine.query(q)
        print(f"\nQ: {q}")
        print(f"A: {resp.answer}")
        print(f"Confidence: {resp.confidence:.2f}")
    else:
        print("No docs in store. Run crawler first.")
