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

    # crawler - basic settings
    max_pages: int = Field(default=100, env="MAX_PAGES")
    crawl_delay: float = Field(default=1.0, env="CRAWL_DELAY")

    # crawler - retry configuration (exponential backoff)
    max_retries: int = Field(
        default=3,
        env="MAX_RETRIES",
        description="Maximum number of retry attempts for failed requests"
    )
    retry_base_delay: float = Field(
        default=1.0,
        env="RETRY_BASE_DELAY",
        description="Base delay in seconds for exponential backoff"
    )
    retry_max_delay: float = Field(
        default=60.0,
        env="RETRY_MAX_DELAY",
        description="Maximum delay in seconds between retries"
    )
    retry_exponential_base: float = Field(
        default=2.0,
        env="RETRY_EXPONENTIAL_BASE",
        description="Base for exponential backoff calculation (delay = base_delay * exponential_base^attempt)"
    )
    retry_jitter: bool = Field(
        default=True,
        env="RETRY_JITTER",
        description="Add random jitter to retry delays to prevent thundering herd"
    )

    # crawler - robots.txt settings
    respect_robots_txt: bool = Field(
        default=True,
        env="RESPECT_ROBOTS_TXT",
        description="Whether to respect robots.txt rules"
    )
    robots_cache_ttl: int = Field(
        default=3600,
        env="ROBOTS_CACHE_TTL",
        description="How long to cache robots.txt files in seconds"
    )

    # chunking
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    # rag
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
