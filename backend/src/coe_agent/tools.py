"""The agent's tools. Each one returns (text for the LLM, list of chunks as artifact).

The artifact travels on the ToolMessage, so the graph knows exactly which passages the
model has seen and can check every citation in the final answer against them.
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.tools import tool

from .config import SOURCES_BY_KEY
from .retrieval import chunks_by_id, load_chunks, search

SourceFilter = Literal["all", "lei", "decreto", "anexos"]

_FILTERS: dict[str, set[str] | None] = {
    "all": None,
    "lei": {"L16642", "L16642-anexos"},
    "decreto": {"D57776", "D57776-anexos"},
    "anexos": {"L16642-anexos", "D57776-anexos"},
}


def format_chunk(c: dict, max_chars: int | None = None) -> str:
    text = c["text"]
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + " [...]"
    title = SOURCES_BY_KEY[c["source"]].title
    heading = f"\nsection: {c['heading']}" if c["heading"] else ""
    return f"<passage id=\"{c['id']}\">\nsource: {title}, {c['locator']}{heading}\n{text}\n</passage>"


@tool(response_format="content_and_artifact")
def search_code(query: str, scope: SourceFilter = "all", k: int = 6) -> tuple[str, list[dict]]:
    """Search the São Paulo Building Code (Lei 16.642/2017), its regulation (Decreto 57.776/2017)
    and their technical annexes. Use Portuguese legal vocabulary in `query` for best results
    (e.g. "pé-direito mínimo dormitório", "largura mínima escada"). `scope` narrows the search:
    "lei", "decreto", "anexos" (technical tables) or "all". Returns passages with citation ids."""
    k = max(1, min(k, 10))
    hits = search(query, k=k, sources=_FILTERS[scope])
    if not hits:
        return "No passages found. Try other Portuguese terms.", []
    chunks = [h.chunk for h in hits]
    return "\n\n".join(format_chunk(c, max_chars=1500) for c in chunks), chunks


@tool(response_format="content_and_artifact")
def read_article(law: Literal["lei", "decreto"], article: str) -> tuple[str, list[dict]]:
    """Read the full text of one article, e.g. law="lei", article="40". Use it when a passage
    refers to another article ("conforme o art. 40 do COE") or when a search hit is truncated."""
    key = "L16642" if law == "lei" else "D57776"
    number = re.sub(r"[^0-9A-Za-z-]", "", article).upper()
    parts = [c for c in load_chunks() if c["source"] == key and (c["article"] or "").upper() == number]
    if not parts:
        return f"Article {article} not found in {SOURCES_BY_KEY[key].title}.", []
    return "\n\n".join(format_chunk(c) for c in parts), parts


@tool(response_format="content_and_artifact")
def read_annex_page(annex: Literal["lei", "decreto"], page: int) -> tuple[str, list[dict]]:
    """Read one full page of a technical annex PDF (tables with minimum dimensions, parking,
    sanitary facilities, fines). Use it to see a whole table when a search hit is cut off."""
    key = "L16642-anexos" if annex == "lei" else "D57776-anexos"
    chunk = chunks_by_id().get(f"{key}:p{page}")
    if chunk is None:
        return f"Page {page} of {SOURCES_BY_KEY[key].title} has no extractable text.", []
    return format_chunk(chunk), [chunk]


TOOLS = [search_code, read_article, read_annex_page]
