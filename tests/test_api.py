import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app
from src.text_processor import TextCleaner, TextChunker, process_documents
from src.crawler import WebCrawler


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_docs():
    return [
        {"content": "Python is a versatile programming language used for web dev and data science.",
         "url": "https://example.com/python", "title": "Python Overview"},
        {"content": "FastAPI is a modern framework for building APIs with Python.",
         "url": "https://example.com/fastapi", "title": "FastAPI Guide"}
    ]


class TestTextCleaner:
    def test_removes_whitespace(self):
        c = TextCleaner()
        result = c.clean_text("Hello    World\n\n\n\nTest")
        assert "    " not in result

    def test_handles_empty(self):
        c = TextCleaner()
        assert c.clean_text("") == ""
        assert c.clean_text(None) == ""

    def test_removes_boilerplate(self):
        c = TextCleaner()
        text = "Main content.\nCopyright © 2024 Company. All rights reserved."
        result = c.remove_boilerplate(text)
        assert "Copyright" not in result
        assert "Main content" in result


class TestTextChunker:
    def test_creates_chunks(self, sample_docs):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(sample_docs[0]["content"],
                                        {"url": sample_docs[0]["url"]})
        assert len(chunks) > 0

    def test_skips_short(self):
        chunker = TextChunker()
        chunks = chunker.chunk_document("Too short", {})
        assert len(chunks) == 0

    def test_token_count(self):
        chunker = TextChunker()
        count = chunker.count_tokens("Hello world")
        assert count > 0


class TestProcessDocs:
    def test_returns_list(self, sample_docs):
        result = process_documents(sample_docs)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_has_metadata(self, sample_docs):
        result = process_documents(sample_docs)
        for c in result:
            assert "content" in c
            assert "metadata" in c


class TestCrawler:
    def test_rejects_invalid_url(self):
        crawler = WebCrawler("https://example.com")
        assert not crawler.is_valid_url("")
        assert not crawler.is_valid_url("not-a-url")
        assert not crawler.is_valid_url("https://other.com/page")

    def test_accepts_valid_url(self):
        crawler = WebCrawler("https://example.com")
        assert crawler.is_valid_url("https://example.com/page")

    def test_rejects_non_html(self):
        crawler = WebCrawler("https://example.com")
        assert not crawler.is_valid_url("https://example.com/img.png")
        assert not crawler.is_valid_url("https://example.com/file.pdf")

    def test_normalizes_relative(self):
        crawler = WebCrawler("https://example.com")
        result = crawler.normalize_url("/page", "https://example.com/docs")
        assert result == "https://example.com/page"

    def test_removes_fragment(self):
        crawler = WebCrawler("https://example.com")
        result = crawler.normalize_url("https://example.com/page#section", "https://example.com")
        assert "#" not in result


class TestAPI:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "name" in r.json()

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "document_count" in data

    def test_stats(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200

    def test_ask_validation(self, client):
        r = client.post("/api/ask", json={"question": ""})
        assert r.status_code == 422

    def test_ask_structure(self, client):
        r = client.post("/api/ask", json={"question": "test?", "n_results": 3})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "sources" in data

    def test_search_validation(self, client):
        r = client.get("/api/search")
        assert r.status_code == 422

    def test_search_works(self, client):
        r = client.get("/api/search?query=test")
        assert r.status_code == 200
        assert "results" in r.json()

    def test_chat_validation(self, client):
        r = client.post("/api/chat", json={"question": "", "chat_history": []})
        assert r.status_code == 422


class TestIntegration:
    @patch("src.vector_store.OpenAI")
    def test_pipeline_mock(self, mock_openai, sample_docs):
        mock_emb = [0.1] * 1536
        mock_openai.return_value.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=mock_emb)]
        )
        chunks = process_documents(sample_docs)
        assert len(chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
