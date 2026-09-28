"""A small FAISS-backed vector store with on-disk persistence."""
import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise so inner product equals cosine similarity."""
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


class VectorStore:
    def __init__(self, embeddings: Embeddings, index_dir: Path):
        self.embeddings = embeddings
        self.index_dir = Path(index_dir)
        self.index: faiss.Index | None = None
        self.docs: list[Document] = []
        self._load()

    # ---------- persistence ----------
    @property
    def _index_file(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def _docs_file(self) -> Path:
        return self.index_dir / "docs.json"

    def _load(self) -> None:
        if self._index_file.exists() and self._docs_file.exists():
            self.index = faiss.read_index(str(self._index_file))
            raw = json.loads(self._docs_file.read_text(encoding="utf-8"))
            self.docs = [Document(page_content=d["text"], metadata=d["meta"]) for d in raw]

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self._index_file))
        raw = [{"text": d.page_content, "meta": d.metadata} for d in self.docs]
        self._docs_file.write_text(json.dumps(raw), encoding="utf-8")

    # ---------- operations ----------
    def add_documents(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        vectors = _normalise(np.array(self.embeddings.embed_documents([d.page_content for d in docs])))
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.docs.extend(docs)
        self.save()
        return len(docs)

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = _normalise(np.array([self.embeddings.embed_query(query)]))
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [(self.docs[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def sources(self) -> list[str]:
        return sorted({d.metadata.get("source", "unknown") for d in self.docs})

    def clear(self) -> None:
        self.index, self.docs = None, []
        for f in (self._index_file, self._docs_file):
            f.unlink(missing_ok=True)
