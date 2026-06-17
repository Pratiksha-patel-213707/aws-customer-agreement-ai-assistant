import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import settings
from .pdf_loader import extract_pdf_pages


@dataclass
class SourceChunk:
    chunk_id: str
    page: int
    text: str
    score: float


class EmbeddingBackend:
    def __init__(self) -> None:
        self.kind = "tfidf"
        self.model = None
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            max_features=20000,
            sublinear_tf=True,
        )
        if settings.embedding_backend == "langchain-huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings

                self.model = HuggingFaceEmbeddings(
                    model_name=settings.embedding_model,
                    model_kwargs={"device": "cpu", "local_files_only": True},
                    encode_kwargs={"normalize_embeddings": True},
                )
                self.kind = "langchain-huggingface"
                self.vectorizer = None
            except Exception:
                self.kind = "tfidf"
                self.vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 3),
                    max_features=20000,
                    sublinear_tf=True,
                )
        elif settings.embedding_backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(settings.embedding_model)
                self.kind = "sentence-transformers"
                self.vectorizer = None
            except Exception:
                self.kind = "tfidf"
                self.vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 3),
                    max_features=20000,
                    sublinear_tf=True,
                )

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self.kind == "langchain-huggingface":
            return np.array(self.model.embed_documents(texts))
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return self.vectorizer.fit_transform(texts).toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.kind == "langchain-huggingface":
            return np.array(self.model.embed_query(texts[0])).reshape(1, -1)
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return self.vectorizer.transform(texts).toarray()

    def embed_query(self, text: str) -> list[float]:
        return self.transform([text])[0].astype(float).tolist()


class RagStore:
    def __init__(self) -> None:
        self.store_dir = settings.vector_store_dir
        self.chunks_path = self.store_dir / "chunks.json"
        self.embeddings_path = self.store_dir / "embeddings.npy"
        self.vectorizer_path = self.store_dir / "tfidf_vectorizer.pkl"
        self.metadata_path = self.store_dir / "metadata.json"
        self.embedding_backend = EmbeddingBackend()
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._load()

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self.embeddings is not None

    def ingest(self, pdf_path: Path | None = None) -> dict:
        source_pdf = pdf_path or settings.pdf_path
        pages = extract_pdf_pages(source_pdf)
        chunks = self._chunk_pages(pages)
        if not chunks:
            raise ValueError(
                "No text could be extracted from the PDF. The file may be blank, corrupted, or image-rendered "
                "in a way OCR cannot read. Replace it with a readable AWS Customer Agreement PDF and retry."
            )

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_backend.fit_transform(texts)

        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        np.save(self.embeddings_path, embeddings)
        if self.embedding_backend.kind == "tfidf":
            with self.vectorizer_path.open("wb") as file:
                pickle.dump(self.embedding_backend.vectorizer, file)
        self.metadata_path.write_text(
            json.dumps(
                {
                    "embedding_backend": self.embedding_backend.kind,
                    "embedding_model": settings.embedding_model,
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self.chunks = chunks
        self.embeddings = embeddings
        return {
            "message": "Document ingested successfully.",
            "pages": len(pages),
            "chunks": len(chunks),
            "embedding_backend": self.embedding_backend.kind,
            "source_pdf": str(source_pdf),
        }

    def retrieve(self, query: str, top_k: int | None = None) -> list[SourceChunk]:
        if not self.ready:
            raise RuntimeError("Document has not been ingested yet.")
        query_embedding = self.embedding_backend.transform([expand_query(query)])
        semantic_scores = cosine_similarity(query_embedding, self.embeddings)[0]
        boosted_scores = _apply_query_boosts(query, semantic_scores, self.chunks)
        candidate_indices = np.argsort(boosted_scores)[::-1][: max(settings.rerank_pool_size, top_k or settings.top_k)]
        reranked = _rerank_candidates(query, candidate_indices, boosted_scores, self.chunks)
        top_indices = _dedupe_indices(reranked, self.chunks)[: top_k or settings.top_k]
        return [
            SourceChunk(
                chunk_id=self.chunks[index]["chunk_id"],
                page=self.chunks[index]["page"],
                text=self.chunks[index]["text"],
                score=float(boosted_scores[index]),
            )
            for index in top_indices
        ]

    def embed_query(self, query: str) -> list[float]:
        return self.embedding_backend.embed_query(query)

    def _load(self) -> None:
        if not self.chunks_path.exists() or not self.embeddings_path.exists() or not self.metadata_path.exists():
            return
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata != {
            "embedding_backend": self.embedding_backend.kind,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }:
            return
        self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(self.embeddings_path)
        if self.vectorizer_path.exists() and self.embedding_backend.kind == "tfidf":
            with self.vectorizer_path.open("rb") as file:
                self.embedding_backend.vectorizer = pickle.load(file)

    def _chunk_pages(self, pages: list[dict]) -> list[dict]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
        chunks: list[dict] = []
        chunk_id = 1
        for page in pages:
            normalized = _normalize_text(page["text"])
            for text in splitter.split_text(normalized):
                if len(text.split()) < 8:
                    continue
                chunks.append(
                    {
                        "chunk_id": f"chunk-{chunk_id}",
                        "page": page["page"],
                        "text": text,
                    }
                )
                chunk_id += 1
        return chunks


def _normalize_text(text: str) -> str:
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def expand_query(query: str) -> str:
    expansions = {
        "password": "password credentials credential account login access key secret authentication security",
        "passwords": "password credentials credential account login access key secret authentication security",
        "term": "term commence effective date remain in effect termination terminate termination date",
        "services covered": "service terms made available third-party content aws services covered scope",
        "covered": "include includes made available service terms third-party content scope",
        "responsibilities": "responsible obligations must will required end users account content security",
        "customer": "you your customer end user account",
    }
    additions = [value for key, value in expansions.items() if key in query.lower()]
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def _apply_query_boosts(query: str, scores: np.ndarray, chunks: list[dict]) -> np.ndarray:
    boosted = scores.copy()
    lowered_query = query.lower()
    wants_service_scope = "services" in lowered_query and any(
        word in lowered_query for word in ["covered", "include", "included", "apply", "scope", "what services"]
    )
    wants_term_definition = "term" in lowered_query and "agreement" in lowered_query

    for index, chunk in enumerate(chunks):
        text = chunk["text"].lower()
        if wants_service_scope and any(
            phrase in text for phrase in [
                "service terms apply",
                "services do not include third-party content",
                "made available",
                "service terms",
            ]
        ):
            boosted[index] += 0.18
        if wants_term_definition and "term of this agreement will commence" in text:
            boosted[index] += 0.25
    return boosted


def _rerank_candidates(query: str, indices: np.ndarray, scores: np.ndarray, chunks: list[dict]) -> list[int]:
    query_terms = _content_terms(expand_query(query))
    reranked: list[tuple[float, int]] = []
    for index in indices:
        text = chunks[index]["text"]
        chunk_terms = _content_terms(text)
        overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
        section_bonus = _section_bonus(query, text)
        final_score = (0.78 * float(scores[index])) + (0.17 * overlap) + section_bonus
        reranked.append((final_score, int(index)))
    return [index for _, index in sorted(reranked, reverse=True)]


def _dedupe_indices(indices: list[int], chunks: list[dict]) -> list[int]:
    selected: list[int] = []
    seen_fingerprints: set[str] = set()
    for index in indices:
        words = re.findall(r"[a-zA-Z]{4,}", chunks[index]["text"].lower())
        fingerprint = " ".join(words[:35])
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        selected.append(index)
    return selected


def _content_terms(text: str) -> set[str]:
    stop_words = {
        "what",
        "which",
        "does",
        "this",
        "that",
        "with",
        "from",
        "about",
        "tell",
        "agreement",
        "customer",
        "amazon",
    }
    return {token for token in re.findall(r"[a-zA-Z]{4,}", text.lower()) if token not in stop_words}


def _section_bonus(query: str, text: str) -> float:
    query_lower = query.lower()
    text_lower = text.lower()
    if "services" in query_lower and any(word in query_lower for word in ["covered", "included", "apply", "scope"]):
        if "service terms apply" in text_lower:
            return 0.18
        if "services do not include third-party content" in text_lower:
            return 0.12
        if "made available" in text_lower and "service terms" in text_lower:
            return 0.10
    if "password" in query_lower or "credential" in query_lower:
        if "log-in credentials and account keys" in text_lower:
            return 0.18
    if "term" in query_lower and "term of this agreement" in text_lower:
        return 0.18
    return 0.0


rag_store = RagStore()


def sources_to_dicts(sources: list[SourceChunk]) -> list[dict]:
    return [asdict(source) for source in sources]
