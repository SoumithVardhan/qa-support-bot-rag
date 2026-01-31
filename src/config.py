import os
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # openai
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    llm_model: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")

    # chroma
    chroma_persist_dir: str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="website_docs", env="COLLECTION_NAME")

    # crawler
    max_pages: int = Field(default=100, env="MAX_PAGES")
    crawl_delay: float = Field(default=1.0, env="CRAWL_DELAY")

    # chunking
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    # rag
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
