from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=8)


class Source(BaseModel):
    chunk_id: str
    page: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    found_answer: bool
    latency_ms: float
    llm_provider: str
    canonical_query: str
    sources: list[Source]


class IngestResponse(BaseModel):
    message: str
    pages: int
    chunks: int
    embedding_backend: str
    source_pdf: str
