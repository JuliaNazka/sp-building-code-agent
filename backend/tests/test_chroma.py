"""The Chroma vector store and the Langfuse switch, with fake embeddings (no API key needed)."""

import hashlib
import math

import pytest
from langchain_core.embeddings import Embeddings

from coe_agent import retrieval, tracing


class BagOfWordsEmbeddings(Embeddings):
    """Hashes normalized words into a fixed-size vector: shared words -> similar vectors."""

    dim = 512

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in retrieval.normalize(text):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture(scope="module")
def chroma_index(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    fake = BagOfWordsEmbeddings()
    mp.setattr(retrieval, "CHROMA_DIR", tmp_path_factory.mktemp("chroma"))
    mp.setattr(retrieval, "_embeddings", lambda: fake)
    retrieval._vector_store.cache_clear()
    retrieval._embed_query_cached.cache_clear()
    retrieval.build_vector_index(batch_size=100)
    yield
    retrieval._vector_store.cache_clear()
    retrieval._embed_query_cached.cache_clear()
    mp.undo()


def test_index_holds_every_chunk(chroma_index):
    assert retrieval.vector_index_ready()
    store = retrieval._vector_store()
    assert store._collection.count() == len(retrieval.load_chunks())


def test_vector_search_finds_a_chunk_by_its_own_text(chroma_index):
    target = retrieval.chunks_by_id()["D57776-anexos:p9"]
    hits = retrieval.search(retrieval._index_text(target), k=3)
    assert hits[0].chunk["id"] == "D57776-anexos:p9"
    assert hits[0].vector_rank == 1


def test_source_filter_is_applied_inside_chroma(chroma_index):
    hits = retrieval.search("vagas para idosos estacionamento", k=6, sources={"L16642-anexos"})
    assert hits and {h.chunk["source"] for h in hits} == {"L16642-anexos"}
    assert any(h.vector_rank for h in hits)


def test_index_built_with_another_model_is_ignored(chroma_index, monkeypatch):
    monkeypatch.setattr(retrieval, "EMBEDDING_MODEL", "models/some-other-model")
    retrieval._vector_store.cache_clear()
    try:
        assert retrieval._vector_store() is None
        hits = retrieval.search("vagas para idosos estacionamento", k=3)
        assert hits and all(h.vector_rank is None for h in hits)  # BM25 only
    finally:
        monkeypatch.undo()
        retrieval._vector_store.cache_clear()


def test_tracing_is_off_without_langfuse_keys(monkeypatch):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)
    assert not tracing.tracing_enabled()
    assert "callbacks" not in tracing.run_config(entrypoint="test")


def test_tracing_adds_the_langfuse_callback_when_configured(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")  # build the handler, send nothing
    from langfuse.langchain import CallbackHandler

    config = tracing.run_config(entrypoint="test")
    assert isinstance(config["callbacks"][0], CallbackHandler)
    assert config["run_name"] == "coe-agent"
