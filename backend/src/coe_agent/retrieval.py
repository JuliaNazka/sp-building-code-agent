"""Hybrid retrieval: BM25 (exact legal terms) + Gemini embeddings (meaning), fused with RRF.

Legal text mixes both kinds of query: "pé-direito mínimo" needs the exact term, while
"how tall must a bedroom be" needs semantic search. Reciprocal Rank Fusion merges the
two ranked lists without having to calibrate their scores against each other.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi

from .config import CHUNKS_FILE, EMBEDDING_MODEL, INDEX_FILE

RRF_K = 60
logger = logging.getLogger(__name__)

STOPWORDS = set(
    """a ao aos as com da das de do dos e em na nas no nos o os ou para pela pelas pelo pelos
    por que se sua suas seu seus um uma qual quais como deve devem ser sobre ate entre mais
    the of and to in is what for""".split()
)


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text) if t not in STOPWORDS]


@dataclass
class Hit:
    chunk: dict
    score: float
    bm25_rank: int | None
    vector_rank: int | None


@lru_cache(maxsize=1)
def load_chunks() -> tuple[dict, ...]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found. Run `coe-ingest` first.")
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f if line.strip())


def chunks_by_id() -> dict[str, dict]:
    return {c["id"]: c for c in load_chunks()}


def _index_text(c: dict) -> str:
    return f"{c['heading']}\n{c['text']}"


@lru_cache(maxsize=1)
def _bm25() -> BM25Okapi:
    return BM25Okapi([normalize(_index_text(c)) for c in load_chunks()])


@lru_cache(maxsize=1)
def _embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _vectors() -> dict[str, list[float]] | None:
    if not INDEX_FILE.exists():
        return None
    data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    if data.get("model") != EMBEDDING_MODEL:
        return None
    return data["vectors"]


def build_vector_index(batch_size: int = 50) -> None:
    import time

    chunks = load_chunks()
    emb = _embeddings()
    vectors: dict[str, list[float]] = {}
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        for attempt in range(5):
            try:
                out = emb.embed_documents([_index_text(c) for c in batch])
                break
            except Exception as exc:  # free tier rate limits: back off and retry
                wait = 15 * (attempt + 1)
                print(f"embedding error ({exc.__class__.__name__}), retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError("embedding failed after 5 attempts")
        vectors.update({c["id"]: v for c, v in zip(batch, out)})
        print(f"embedded {len(vectors)}/{len(chunks)}")
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps({"model": EMBEDDING_MODEL, "vectors": vectors}), encoding="utf-8")
    _vectors.cache_clear()
    print(f"wrote vector index to {INDEX_FILE}")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@lru_cache(maxsize=256)
def _embed_query_cached(query: str) -> tuple[float, ...]:
    return tuple(_embeddings().embed_query(query))


def _embed_query(query: str) -> tuple[float, ...] | None:
    """Embed the query; on API errors (e.g. free-tier rate limit) fall back to BM25 only."""
    try:
        return _embed_query_cached(query)
    except Exception as exc:
        logger.warning("query embedding failed (%s); using BM25 only", exc.__class__.__name__)
        return None


def search(query: str, k: int = 6, sources: set[str] | None = None, pool: int = 30) -> list[Hit]:
    """Return the top-k chunks for `query`, optionally restricted to some source keys."""
    chunks = load_chunks()
    allowed = [i for i, c in enumerate(chunks) if not sources or c["source"] in sources]

    bm25_scores = _bm25().get_scores(normalize(query))
    bm25_order = sorted(allowed, key=lambda i: bm25_scores[i], reverse=True)[:pool]
    bm25_order = [i for i in bm25_order if bm25_scores[i] > 0]

    vector_order: list[int] = []
    vectors = _vectors()
    q = _embed_query(query) if vectors is not None else None
    if q is not None:
        scored = [(i, _cosine(q, vectors[chunks[i]["id"]])) for i in allowed if chunks[i]["id"] in vectors]
        vector_order = [i for i, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:pool]]

    fused: dict[int, float] = {}
    for order in (bm25_order, vector_order):
        for rank, i in enumerate(order):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

    bm25_rank = {i: r + 1 for r, i in enumerate(bm25_order)}
    vector_rank = {i: r + 1 for r, i in enumerate(vector_order)}
    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [Hit(chunks[i], fused[i], bm25_rank.get(i), vector_rank.get(i)) for i in best]


def vector_index_ready() -> bool:
    return _vectors() is not None
