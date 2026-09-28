"""FastAPI application exposing upload + chat endpoints."""
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config
from app.ingest import ingest_pdf
from app.rag import RAGAssistant

app = FastAPI(title="AI Document Assistant (RAG)")
_assistant: RAGAssistant | None = None
_lock = threading.Lock()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_assistant() -> RAGAssistant:
    """Create the assistant lazily so the app can start without an API key."""
    global _assistant
    if _assistant is None:
        if not config.OPENAI_API_KEY:
            raise HTTPException(500, "OPENAI_API_KEY is not set. Copy .env.example to .env.")
        _assistant = RAGAssistant()
    return _assistant


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    dest = config.UPLOAD_DIR / Path(file.filename).name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    if dest.stat().st_size > config.MAX_UPLOAD_MB * 1024 * 1024:
        dest.unlink()
        raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB.")
    try:
        chunks = ingest_pdf(dest)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, str(exc))
    with _lock:
        added = get_assistant().store.add_documents(chunks)
    return {"file": dest.name, "chunks_indexed": added}


@app.get("/documents")
def documents():
    return {"documents": get_assistant().store.sources()}


@app.post("/chat")
def chat(req: ChatRequest):
    """Non-streaming answer with source citations."""
    return get_assistant().ask(req.question, req.session_id)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Token-by-token streamed answer."""
    return StreamingResponse(
        get_assistant().stream(req.question, req.session_id),
        media_type="text/plain; charset=utf-8",
    )


@app.post("/reset")
def reset(req: ChatRequest):
    get_assistant().reset(req.session_id)
    return {"status": "history cleared"}
