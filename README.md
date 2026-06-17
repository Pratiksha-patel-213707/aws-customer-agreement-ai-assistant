# AWS Customer Agreement AI Assistant

A lightweight RAG-based assistant for querying the AWS Customer Agreement PDF. The project combines a FastAPI backend, local retrieval logic, optional LLM generation, and a small React frontend for chat and analytics.

## Overview

This application lets you:

- ingest a PDF agreement,
- chunk and embed the document,
- retrieve the most relevant passages for a question, and
- return a grounded answer with source references.

It is designed for document-grounded Q&A.

## Main Features

- **Document ingestion** with text extraction and OCR fallback.
- **Chunk-based retrieval** using embeddings and cosine similarity.
- **Answer generation** with provider fallback when model calls are unavailable.
- **Analytics logging** for frequent questions, no-answer cases, latency, and query volume.
- **Simple frontend dashboard** to ask questions and inspect results.

## Architecture

- `POST /ingest` extracts text from `AWS Customer Agreement.pdf`, uses OCR when the PDF has no selectable text, chunks the document, creates embeddings, and stores the vector metadata under `data/vector_store`.
- `POST /ask` embeds the user's question, retrieves relevant chunks, generates a grounded answer, returns the answer plus source references, and logs the interaction to SQLite.
- `GET /analytics` reads usage logs to report frequent questions, no-answer queries, latency, and daily activity.
- `frontend/` is a Vite React app that calls the FastAPI backend.

## Design Notes

- Chunking uses LangChain `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200` to preserve clause context while keeping sections readable.
- Embeddings default to the Hugging Face backend (`BAAI/bge-small-en-v1.5`), while the repository also supports a TF-IDF fallback for environments without model downloads.
- Retrieval uses a reranking workflow so the most relevant evidence is prioritized before answer generation.
- The query pipeline expands a few common legal phrases to improve intent matching without changing the document content.
- SQLite stores query text, canonical grouping data, answers, top similarity scores, provider details, and timestamps.

## Prerequisites

- Python 3.10+
- Node.js and npm
- A readable copy of `AWS Customer Agreement.pdf`
- Optional API keys for Gemini or Hugging Face

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Configure one of the following providers in `.env`:

```bash
HUGGINGFACE_API_KEY=your_key_here
LLM_PROVIDER=huggingface
```

or:

```bash
GEMINI_API_KEY=your_key_here
LLM_PROVIDER=gemini
```

## Run the Backend

```bash
.\.venv\Scripts\activate
uvicorn backend.app.main:app --reload
```

Open the API docs at `http://localhost:8000/docs`.

To ingest the PDF:

```bash
curl -X POST http://localhost:8000/ingest
```

If ingestion reports that no text could be extracted, replace the PDF with a readable version and run `/ingest` again. The pipeline supports selectable text PDFs and scanned PDFs when OCR is available.

## Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Seed Analytics Data

After the backend is running, you can generate sample logs:

```bash
.\.venv\Scripts\python.exe backend\seed_queries.py
```

You can also refresh analytics directly from the API:

```bash
curl http://localhost:8000/analytics
```

## Project Structure

```text
backend/
  app/
    config.py       environment settings
    database.py     SQLite schema, logging, analytics SQL
    llm.py          answer generation and fallback logic
    main.py         FastAPI routes
    pdf_loader.py   PDF text extraction and OCR fallback
    rag.py          chunking, embeddings, retrieval, reranking
frontend/
  src/
    main.jsx        React chat and analytics dashboard
    styles.css      UI styles
```

## Demo Checklist

1. Start the FastAPI server.
2. Start the React frontend.
3. Run the ingestion step.
4. Ask questions such as "When can AWS suspend services?"
5. Ask out-of-scope questions such as "Does this document explain sourdough bread?"
6. Review analytics for frequent questions, no-answer cases, and response latency.
