import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .config import settings


def _database_path() -> Path:
    parsed = urlparse(settings.database_url)
    if parsed.scheme != "sqlite":
        raise ValueError("Only sqlite DATABASE_URL values are supported by this demo project.")
    raw_path = parsed.path.lstrip("/") if parsed.netloc == "" else f"{parsed.netloc}{parsed.path}"
    return Path(raw_path)


DB_PATH = _database_path()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                original_query TEXT,
                canonical_query TEXT,
                query_embedding TEXT,
                answer TEXT NOT NULL,
                found_answer INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                top_similarity REAL,
                source_count INTEGER NOT NULL,
                llm_provider TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS question_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_query TEXT NOT NULL UNIQUE,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_column(conn, "query_logs", "original_query", "TEXT")
        _ensure_column(conn, "query_logs", "canonical_query", "TEXT")
        _ensure_column(conn, "query_logs", "query_embedding", "TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_query_logs_question ON query_logs(question)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_query_logs_canonical ON query_logs(canonical_query)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_query_logs_found ON query_logs(found_answer)")


@contextmanager
def get_connection():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_query(
    *,
    original_query: str,
    canonical_query: str,
    query_embedding: list[float],
    answer: str,
    found_answer: bool,
    latency_ms: float,
    top_similarity: float | None,
    source_count: int,
    llm_provider: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO query_logs
                (
                    question, original_query, canonical_query, query_embedding, answer, found_answer,
                    latency_ms, top_similarity, source_count, llm_provider
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                canonical_query,
                original_query,
                canonical_query,
                json.dumps(query_embedding),
                answer,
                int(found_answer),
                latency_ms,
                top_similarity,
                source_count,
                llm_provider,
            ),
        )


def analytics() -> dict:
    with get_connection() as conn:
        frequent = conn.execute(
            """
            SELECT COALESCE(canonical_query, question) AS question, COUNT(*) AS count, AVG(latency_ms) AS avg_latency_ms
            FROM query_logs
            GROUP BY LOWER(TRIM(COALESCE(canonical_query, question)))
            ORDER BY count DESC, MAX(created_at) DESC
            LIMIT 10
            """
        ).fetchall()
        no_answer = conn.execute(
            """
            SELECT
                COALESCE(original_query, question) AS original_query,
                COALESCE(canonical_query, question) AS canonical_query,
                answer,
                created_at,
                top_similarity
            FROM query_logs
            WHERE found_answer = 0
            ORDER BY created_at DESC
            LIMIT 20
            """
        ).fetchall()
        summary = conn.execute(
            """
            SELECT
                COUNT(*) AS total_queries,
                COALESCE(AVG(latency_ms), 0) AS average_latency_ms,
                SUM(CASE WHEN found_answer = 0 THEN 1 ELSE 0 END) AS no_answer_count,
                SUM(CASE WHEN found_answer = 1 THEN 1 ELSE 0 END) AS answered_count
            FROM query_logs
            """
        ).fetchone()
        by_day = conn.execute(
            """
            SELECT DATE(created_at) AS date, COUNT(*) AS count, AVG(latency_ms) AS avg_latency_ms
            FROM query_logs
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT 14
            """
        ).fetchall()

    return {
        "summary": dict(summary),
        "most_frequent_questions": [dict(row) for row in frequent],
        "no_answer_queries": [dict(row) for row in no_answer],
        "queries_by_day": [dict(row) for row in by_day],
    }


def get_or_create_canonical_query(original_query: str, embedding: list[float], threshold: float) -> str:
    normalized_embedding = _normalize_embedding(embedding)
    with get_connection() as conn:
        rows = conn.execute("SELECT canonical_query, embedding FROM question_groups").fetchall()
        best_query = None
        best_score = -1.0
        for row in rows:
            stored_embedding = json.loads(row["embedding"])
            score = _cosine_similarity(normalized_embedding, stored_embedding)
            if score > best_score:
                best_score = score
                best_query = row["canonical_query"]

        if best_query and best_score >= threshold:
            conn.execute(
                "UPDATE question_groups SET updated_at = CURRENT_TIMESTAMP WHERE canonical_query = ?",
                (best_query,),
            )
            return best_query

        canonical_query = original_query.strip()
        conn.execute(
            """
            INSERT OR IGNORE INTO question_groups (canonical_query, embedding)
            VALUES (?, ?)
            """,
            (canonical_query, json.dumps(normalized_embedding)),
        )
        return canonical_query


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_type: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def _normalize_embedding(embedding: list[float]) -> list[float]:
    vector = np.array(embedding, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    left_vector = np.array(left, dtype=float)
    right_vector = np.array(right, dtype=float)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator == 0:
        return -1.0
    return float(np.dot(left_vector, right_vector) / denominator)
