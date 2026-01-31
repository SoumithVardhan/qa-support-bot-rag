import logging
from typing import List, Dict, Any
import hashlib

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI
from tqdm import tqdm

from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, model=None, api_key=None):
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        self.client = OpenAI(api_key=self.api_key)

    def generate_embedding(self, text):
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def generate_embeddings(self, texts, batch_size=100):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([item.embedding for item in resp.data])
        return embeddings


class VectorStore:
    def __init__(self, collection_name=None, persist_dir=None, embedding_gen=None):
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_dir or settings.chroma_persist_dir

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_gen = embedding_gen or EmbeddingGenerator()

    def _make_id(self, content, metadata):
        s = f"{metadata.get('url', '')}{content[:100]}"
        return hashlib.md5(s.encode()).hexdigest()

    def add_documents(self, chunks, batch_size=100):
        if not chunks:
            logger.warning("No chunks to add")
            return 0

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        docs = [c["content"] for c in chunks]
        metas = [c.get("metadata", {}) for c in chunks]
        ids = [c.get("chunk_id") or self._make_id(c["content"], c.get("metadata", {}))
               for c in chunks]

        embeddings = self.embedding_gen.generate_embeddings(docs, batch_size)

        for i in range(0, len(docs), batch_size):
            end = min(i + batch_size, len(docs))
            self.collection.add(
                documents=docs[i:end],
                embeddings=embeddings[i:end],
                metadatas=metas[i:end],
                ids=ids[i:end]
            )

        logger.info(f"Added {len(chunks)} chunks")
        return len(chunks)

    def search(self, query, n_results=None, filter_meta=None):
        n_results = n_results or settings.top_k_results
        query_emb = self.embedding_gen.generate_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            where=filter_meta,
            include=["documents", "metadatas", "distances"]
        )

        formatted = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                dist = results["distances"][0][i] if results["distances"] else 0
                formatted.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": dist,
                    "relevance_score": 1 - dist
                })
        return formatted

    def get_collection_stats(self):
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.persist_directory
        }

    def clear_collection(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Cleared collection: {self.collection_name}")

    def delete_by_url(self, url):
        self.collection.delete(where={"url": url})
        logger.info(f"Deleted chunks from: {url}")


def create_vector_store(chunks, collection_name=None, clear_existing=False):
    store = VectorStore(collection_name=collection_name)
    if clear_existing:
        store.clear_collection()
    store.add_documents(chunks)
    return store


if __name__ == "__main__":
    test_chunks = [
        {"content": "Python is a versatile programming language.",
         "metadata": {"url": "https://example.com/python", "title": "Python"},
         "chunk_id": "python_1"},
        {"content": "FastAPI is a modern web framework for building APIs.",
         "metadata": {"url": "https://example.com/fastapi", "title": "FastAPI"},
         "chunk_id": "fastapi_1"},
    ]

    store = create_vector_store(test_chunks, collection_name="test", clear_existing=True)

    print("\nSearching for 'web framework':")
    results = store.search("web framework for APIs", n_results=2)
    for r in results:
        print(f"  Score: {r['relevance_score']:.3f} - {r['content'][:60]}...")

    print("\nStats:", store.get_collection_stats())
