# AI Document Assistant (RAG)

A Retrieval-Augmented Generation (RAG) app that answers questions about uploaded PDFs.
Answers are grounded in the document text and cite the file and page they came from.

## Features
- PDF ingestion pipeline: text extraction, cleaning and overlapping chunking
- Semantic search using OpenAI embeddings and a persistent **FAISS** vector index
- **LangChain** components for text splitting, embeddings and the chat model
- **FastAPI** backend with upload, chat and real-time streaming endpoints
- Conversational memory per session, so follow-up questions keep context
- Prompt that restricts answers to retrieved context to reduce hallucinations
- Minimal browser chat UI and offline unit tests (no API key needed)

## Architecture
```
PDF -> extract & clean -> chunk -> embeddings -> FAISS index
                                                   |
Question -> embed -> top-k similar chunks ---------+
                          |
              grounded prompt + chat history -> LLM -> streamed answer + sources
```

## Project structure
```
app/
  config.py        settings from environment variables
  ingest.py        PDF loading, cleaning, chunking
  vector_store.py  FAISS store with save/load
  rag.py           retrieval, prompt, memory, streaming
  main.py          FastAPI endpoints
static/index.html  chat UI
tests/             offline tests (fake embeddings + fake LLM)
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your OPENAI_API_KEY
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000, upload a PDF and start asking questions.
Interactive API docs: http://127.0.0.1:8000/docs

## API
| Method | Endpoint       | Description                                   |
|--------|----------------|-----------------------------------------------|
| POST   | `/upload`      | Upload and index a PDF (multipart `file`)     |
| POST   | `/chat`        | Answer + sources as JSON                      |
| POST   | `/chat/stream` | Token-by-token streamed answer                |
| GET    | `/documents`   | List indexed documents                        |
| POST   | `/reset`       | Clear a session's chat history                |
| GET    | `/health`      | Health check                                  |

Example:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "session_id": "demo"}'
```

## Tests
```bash
pytest -q
```

## Limitations
- Scanned/image-only PDFs need OCR first (not included)
- Chat history is stored in memory and resets when the server restarts
- Uses a flat FAISS index, which is fine for small/medium document sets
