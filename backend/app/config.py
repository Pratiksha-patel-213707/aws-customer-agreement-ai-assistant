from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AWS Customer Agreement AI Assistant"
    database_url: str = "sqlite:///data/usage.db"
    vector_store_dir: Path = Path("data/vector_store")
    pdf_path: Path = Path("AWS Customer Agreement.pdf")
    llm_provider: str = "huggingface"
    huggingface_api_key: str | None = None
    huggingface_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"
    embedding_backend: str = "langchain-huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    frontend_origin: str = "http://localhost:5173"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 6
    min_similarity: float = 0.025
    rerank_pool_size: int = 14
    question_group_similarity_threshold: float = 0.85

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
