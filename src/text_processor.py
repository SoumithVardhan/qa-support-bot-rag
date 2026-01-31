import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    token_count: int


class TextCleaner:
    @staticmethod
    def clean_text(text):
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        # text = re.sub(r'http[s]?://\S+', '', text)  # uncomment to strip urls
        return text.strip()

    @staticmethod
    def remove_boilerplate(text):
        patterns = [
            r"(?i)copyright\s*©?\s*\d{4}.*?(?=\n|$)",
            r"(?i)all rights reserved.*?(?=\n|$)",
            r"(?i)privacy policy.*?(?=\n|$)",
            r"(?i)terms of service.*?(?=\n|$)",
            r"(?i)cookie policy.*?(?=\n|$)",
            r"(?i)subscribe to our newsletter.*?(?=\n|$)",
        ]
        for p in patterns:
            text = re.sub(p, "", text)
        return text.strip()


class TextChunker:
    def __init__(self, chunk_size=None, chunk_overlap=None, model_name=None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.model_name = model_name or settings.embedding_model

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def chunk_document(self, content, metadata=None):
        metadata = metadata or {}

        cleaner = TextCleaner()
        cleaned = cleaner.clean_text(content)
        cleaned = cleaner.remove_boilerplate(cleaned)

        if len(cleaned) < 50:
            return []

        texts = self.splitter.split_text(cleaned)

        chunks = []
        for i, text in enumerate(texts):
            chunk_id = f"{metadata.get('url', 'doc')}_{i}"
            chunks.append(TextChunk(
                content=text,
                metadata={**metadata, "chunk_index": i, "total_chunks": len(texts)},
                chunk_id=chunk_id,
                token_count=self.count_tokens(text)
            ))
        return chunks

    def chunk_documents(self, documents):
        all_chunks = []
        for doc in documents:
            content = doc.get("content", "")
            meta = {
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "source": doc.get("source", "web")
            }
            chunks = self.chunk_document(content, meta)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} docs")
        return all_chunks


def process_documents(documents):
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)
    return [
        {"content": c.content, "metadata": c.metadata,
         "chunk_id": c.chunk_id, "token_count": c.token_count}
        for c in chunks
    ]


if __name__ == "__main__":
    sample = """
    Welcome to Our Documentation

    This is a guide to using our product.
    It contains instructions and examples.

    Getting Started

    To begin, install the software:
    1. Download the installer
    2. Run the wizard
    3. Configure settings

    Advanced Features

    Includes automation and scripting.

    Copyright © 2024 Example Corp. All rights reserved.
    """

    docs = [{"content": sample, "url": "https://example.com/docs", "title": "Docs"}]
    chunks = process_documents(docs)
    print(f"Created {len(chunks)} chunks:")
    for c in chunks:
        print(f"  - {c['chunk_id']}: {len(c['content'])} chars, {c['token_count']} tokens")
