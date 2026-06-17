# AWS Customer Agreement AI Assistant

A lightweight RAG-based assistant for querying the AWS Customer Agreement PDF. The project combines a FastAPI backend, retrieval logic, optional LLM generation, and a small React frontend for chat and analytics.

## 1. Overview

This application lets you:

- ingest a PDF agreement,
- chunk and embed the document,
- retrieve the most relevant passages for a question, and
- return a grounded answer with source references.

It is designed for document-grounded Q&A over legal text.

## 2. Features

- **Document ingestion** with text extraction and OCR fallback.
- **Chunk-based retrieval** using embeddings and cosine similarity.
- **Answer generation** with provider fallback when model calls are unavailable.
- **Analytics logging** for frequent questions, no-answer cases, latency, and query volume.
- **Frontend dashboard** to ask questions and inspect results.

## 3. Architecture Overview

The project is split into three main layers:

- **Frontend**: React/Vite app that lets the user ask questions and view analytics.
- **Backend**: FastAPI service that exposes `/ingest`, `/ask`, and `/analytics`.
- **Retrieval + LLM pipeline**: loads the PDF, chunks it, embeds it, retrieves evidence, and produces a final answer.

A simple flow is:

1. The user submits a question from the frontend.
2. The backend checks whether the document has already been ingested.
3. If needed, the PDF is extracted, split into chunks, and embeddings are stored.
4. The retrieval layer scores the relevant chunks.
5. The answer layer generates a concise answer grounded in the retrieved evidence.
6. The interaction is logged so analytics can show query patterns.

For a more detailed architecture explanation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 4. Design Decisions and Assumptions

- **Chunking strategy**: `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200` to preserve clause context while keeping sections readable.
- **Top-k retrieval**: the app retrieves a small set of relevant sections (`top_k=6` by default) and expands the candidate pool before reranking.
- **Embedding choice**: the project prefers Hugging Face embeddings when available, but falls back to TF-IDF if embeddings cannot be downloaded.
- **Answering behavior**: the response pipeline favors direct, synthesized answers over long copied text.
- **Analytics model**: usage logs are stored in SQLite so the dashboard can show recent queries, latency, and common patterns.
- **Assumption**: the PDF content is readable enough for extraction; OCR is used when needed.

## 5. Prerequisites

- Python 3.10+
- Node.js and npm
- A readable copy of `AWS Customer Agreement.pdf`
- Optional API keys for Hugging Face or Gemini

## 6. End-to-End Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/Pratiksha-patel-213707/aws-customer-agreement-ai-assistant.git
cd aws-customer-agreement-ai-assistant
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create environment variables

Copy the example file:

```bash
copy .env.example .env
```

Then choose one provider configuration:

For Hugging Face:

```env
LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=your_key_here
```

For Gemini:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

If you do not configure a provider key, the app still runs with fallback behavior for retrieval and local answer heuristics.

### Step 5: Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## 7. Run the Application

### Start the backend

```bash
uvicorn backend.app.main:app --reload
```

The API will be available at:

- http://localhost:8000/docs

### Start the frontend

In another terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at:

- http://localhost:5173

### Ingest the document

Once the backend is running:

```bash
curl -X POST http://localhost:8000/ingest
```

If ingestion reports that no text could be extracted, replace the PDF with a readable version and run the endpoint again.

## 8. Demo / Testing Flow

You can generate sample questions for analytics with:

```bash
python backend/seed_queries.py
```

You can also inspect analytics directly:

```bash
curl http://localhost:8000/analytics
```

Suggested demo questions:

- “What services are covered by the agreement?”
- “When does the agreement term begin?”
- “Does this document explain sourdough bread?”

## 9. Project Structure

```text
backend/
  app/
    config.py       environment settings
    database.py     SQLite schema and analytics logic
    llm.py          answer generation and fallback logic
    main.py         FastAPI routes
    pdf_loader.py   PDF text extraction and OCR fallback
    rag.py          chunking, embeddings, retrieval, reranking
frontend/
  src/
    main.jsx        React chat and analytics dashboard
    styles.css      UI styles
docs/
  ARCHITECTURE.md   architecture notes
```

## 10. Submission Notes

- The repository includes the full source code for both backend and frontend.
- The README is intended to let someone clone the repo and follow the setup steps without needing extra instructions.
- The architecture details are documented separately in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- A short demo video/GIF should be added to the repository or linked from the README to show the app running and the analytics dashboard being used.
