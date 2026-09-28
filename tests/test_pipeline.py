"""Offline tests: no API key needed (uses fake embeddings and a fake LLM)."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.ingest import clean_text, split_documents
from app.rag import RAGAssistant
from app.vector_store import VectorStore


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def stream(self, messages):
        yield FakeChunk("Hello ")
        yield FakeChunk("world")

    def invoke(self, messages):
        return FakeChunk("Hello world")


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("data-\nbase   systems") == "database systems"


def test_split_documents_keeps_metadata():
    doc = Document(page_content="word " * 1000, metadata={"source": "a.pdf", "page": 1})
    chunks = split_documents([doc])
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "a.pdf" for c in chunks)


def test_vector_store_search_and_persistence(tmp_path):
    emb = DeterministicFakeEmbedding(size=32)
    store = VectorStore(emb, tmp_path)
    store.add_documents([
        Document(page_content="solar panels", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="home security", metadata={"source": "a.pdf", "page": 2}),
    ])
    top, _ = store.search("solar panels", k=1)[0]
    assert top.page_content == "solar panels"

    reloaded = VectorStore(emb, tmp_path)            # index survives restart
    assert len(reloaded.docs) == 2


def test_assistant_streams_and_remembers(tmp_path):
    emb = DeterministicFakeEmbedding(size=32)
    store = VectorStore(emb, tmp_path)
    store.add_documents([Document(page_content="abc", metadata={"source": "a.pdf", "page": 1})])
    bot = RAGAssistant(store=store, embeddings=emb, llm=FakeLLM())

    assert "".join(bot.stream("hi", "s1")) == "Hello world"
    assert len(bot.history["s1"]) == 2                # question + answer stored
    assert bot.ask("again", "s1")["sources"][0]["source"] == "a.pdf"


def test_empty_index_message(tmp_path):
    emb = DeterministicFakeEmbedding(size=32)
    bot = RAGAssistant(store=VectorStore(emb, tmp_path), embeddings=emb, llm=FakeLLM())
    assert "Upload a PDF" in bot.ask("anything")["answer"]
