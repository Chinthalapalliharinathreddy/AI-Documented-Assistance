"""RAG pipeline: retrieve -> build grounded prompt -> stream LLM answer."""
from collections import defaultdict
from typing import Iterator

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app import config
from app.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the question using ONLY the context "
    "provided below. If the answer is not in the context, reply exactly: "
    "\"I couldn't find that in the uploaded documents.\" Do not use outside "
    "knowledge or guess. Be concise and cite sources as (file, page).\n\n"
    "Context:\n{context}"
)


def format_context(results) -> str:
    return "\n\n".join(
        f"[{d.metadata['source']}, page {d.metadata['page']}]\n{d.page_content}"
        for d, _ in results
    )


def unique_sources(results) -> list[dict]:
    seen, out = set(), []
    for d, score in results:
        key = (d.metadata["source"], d.metadata["page"])
        if key not in seen:
            seen.add(key)
            out.append({"source": key[0], "page": key[1], "score": round(score, 3)})
    return out


class RAGAssistant:
    def __init__(self, store: VectorStore | None = None, embeddings: Embeddings | None = None, llm=None):
        embeddings = embeddings or OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.store = store or VectorStore(embeddings, config.INDEX_DIR)
        self.llm = llm or ChatOpenAI(model=config.CHAT_MODEL, temperature=0, streaming=True)
        # conversational memory: session_id -> [(role, text), ...]
        self.history: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def _messages(self, question: str, session_id: str, results):
        messages = [("system", SYSTEM_PROMPT.format(context=format_context(results)))]
        messages += self.history[session_id][-2 * config.MAX_HISTORY_TURNS:]
        messages.append(("human", question))
        return messages

    def _retrieval_query(self, question: str, session_id: str) -> str:
        """Fold the previous user turn in so follow-ups like 'and its cost?' still retrieve well."""
        prior = [t for role, t in self.history[session_id] if role == "human"]
        return f"{prior[-1]} {question}" if prior else question

    def stream(self, question: str, session_id: str = "default") -> Iterator[str]:
        results = self.store.search(self._retrieval_query(question, session_id), k=config.TOP_K)
        if not results:
            yield "No documents indexed yet. Upload a PDF first."
            return
        answer = ""
        for chunk in self.llm.stream(self._messages(question, session_id, results)):
            token = chunk.content or ""
            answer += token
            yield token
        self.history[session_id] += [("human", question), ("ai", answer)]

    def ask(self, question: str, session_id: str = "default") -> dict:
        results = self.store.search(self._retrieval_query(question, session_id), k=config.TOP_K)
        if not results:
            return {"answer": "No documents indexed yet. Upload a PDF first.", "sources": []}
        answer = self.llm.invoke(self._messages(question, session_id, results)).content
        self.history[session_id] += [("human", question), ("ai", answer)]
        return {"answer": answer, "sources": unique_sources(results)}

    def reset(self, session_id: str) -> None:
        self.history.pop(session_id, None)
