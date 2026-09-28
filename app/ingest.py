"""PDF ingestion: extract -> clean -> chunk."""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app import config


def clean_text(text: str) -> str:
    """Normalise whitespace and fix words hyphenated across line breaks."""
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # re-join hyphenated words
    text = re.sub(r"[ \t]+", " ", text)            # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse blank lines
    return text.strip()


def load_pdf(path: Path) -> list[Document]:
    """Read a PDF and return one Document per non-empty page."""
    reader = PdfReader(str(path))
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_number},
                )
            )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks so retrieval stays precise."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_pdf(path: Path) -> list[Document]:
    """Full pipeline for one PDF: extract, clean and chunk."""
    docs = load_pdf(path)
    if not docs:
        raise ValueError("No extractable text found (scanned PDFs need OCR).")
    return split_documents(docs)
