"""Tests that run without an API key (BM25 only, no LLM)."""

import re

import pytest

from coe_agent import retrieval
from coe_agent.graph import find_citations
from coe_agent.server import to_langchain
from coe_agent.tools import read_annex_page, read_article, search_code


@pytest.fixture(autouse=True)
def bm25_only(monkeypatch):
    monkeypatch.setattr(retrieval, "_vectors", lambda: None)


def test_chunk_ids_are_unique_and_articles_complete():
    chunks = retrieval.load_chunks()
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for key, last in (("L16642", 123), ("D57776", 112)):
        numbers = {int(re.match(r"\d+", c["article"]).group()) for c in chunks if c["source"] == key}
        assert numbers == set(range(1, last + 1))


def test_revoked_text_is_dropped():
    art16 = " ".join(c["text"] for c in retrieval.load_chunks() if c["id"].startswith("D57776:art-16"))
    assert "autarquias universitárias, independe" not in art16


@pytest.mark.parametrize(
    "query, expected",
    [
        ("vagas para idosos estacionamento", "L16642-anexos:p5"),
        ("pé-direito dimensionamento mínimo compartimentos", "D57776-anexos:p9"),
        ("tabela de multas obra sem alvará", "L16642-anexos:p12"),
    ],
)
def test_bm25_finds_the_right_annex_page(query, expected):
    top = [h.chunk["id"] for h in retrieval.search(query, k=5)]
    assert expected in top


def test_tools_return_artifacts_for_citation_check():
    msg = search_code.invoke({"type": "tool_call", "id": "1", "name": "search_code", "args": {"query": "recuo"}})
    assert msg.artifact and all("id" in c for c in msg.artifact)
    assert all(f'id="{c["id"]}"' in msg.content for c in msg.artifact)

    art = read_article.invoke({"type": "tool_call", "id": "2", "name": "read_article", "args": {"law": "lei", "article": "3º"}})
    assert art.artifact[0]["id"].startswith("L16642:art-3")

    page = read_annex_page.invoke({"type": "tool_call", "id": "3", "name": "read_annex_page", "args": {"annex": "decreto", "page": 9}})
    assert "Dimensionamento mínimo" in page.content


def test_citation_regex():
    text = "Mínimo 2,50 m [D57776-anexos:p9]; ver também [L16642:art-3-4] e [nota]."
    assert find_citations(text) == ["D57776-anexos:p9", "L16642:art-3-4"]


def test_ui_messages_to_langchain():
    history, question = to_langchain(
        [
            {"role": "user", "parts": [{"type": "text", "text": "oi"}]},
            {"role": "assistant", "parts": [{"type": "dynamic-tool"}, {"type": "text", "text": "olá"}]},
            {"role": "user", "parts": [{"type": "text", "text": "pé-direito?"}]},
        ]
    )
    assert [m.type for m in history] == ["human", "ai"]
    assert question == "pé-direito?"


def test_citation_regex_tolerates_inciso_inside_brackets():
    assert find_citations("ver [L16642:art-15, II] e [L16642:art-23-1, II]") == ["L16642:art-15", "L16642:art-23-1"]
    assert find_citations("[L16642:art-12-1, D57776:art-71] e [D57776:art-71]") == ["L16642:art-12-1", "D57776:art-71"]


def test_search_falls_back_to_bm25_when_embedding_fails(monkeypatch):
    monkeypatch.setattr(retrieval, "_vectors", lambda: {"x": [1.0]})

    def boom(_q):
        raise RuntimeError("429")

    monkeypatch.setattr(retrieval, "_embed_query_cached", boom)
    hits = retrieval.search("vagas para idosos estacionamento", k=3)
    assert hits and all(h.vector_rank is None for h in hits)
