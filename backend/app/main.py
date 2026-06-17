import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import analytics, get_or_create_canonical_query, init_db, log_query
from .llm import NO_ANSWER, answer_question
from .rag import rag_store, sources_to_dicts
from .schemas import AskRequest, AskResponse, IngestResponse


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "document_ready": rag_store.ready}


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> dict:
    try:
        return rag_store.ingest()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> dict:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not rag_store.ready:
        raise HTTPException(status_code=409, detail="Document is not ingested yet. Call POST /ingest first.")

    started = time.perf_counter()
    query_embedding = rag_store.embed_query(query)
    canonical_query = get_or_create_canonical_query(
        query,
        query_embedding,
        settings.question_group_similarity_threshold,
    )
    sources = rag_store.retrieve(query, payload.top_k)
    answer, provider = answer_question(query, sources)
    found_answer = NO_ANSWER.lower() not in answer.lower()
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    top_similarity = sources[0].score if sources else None

    log_query(
        original_query=query,
        canonical_query=canonical_query,
        query_embedding=query_embedding,
        answer=answer,
        found_answer=found_answer,
        latency_ms=latency_ms,
        top_similarity=top_similarity,
        source_count=len(sources),
        llm_provider=provider,
    )

    return {
        "answer": answer,
        "found_answer": found_answer,
        "latency_ms": latency_ms,
        "llm_provider": provider,
        "canonical_query": canonical_query,
        "sources": sources_to_dicts(sources),
    }


@app.get("/analytics")
def get_analytics() -> dict:
    return analytics()
