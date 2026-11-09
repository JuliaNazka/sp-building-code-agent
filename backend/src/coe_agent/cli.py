"""Ask the agent from the terminal.

    coe-ask "Qual o pé-direito mínimo de um dormitório?"
    coe-ask --trace "What is the minimum width of a residential staircase?"
    coe-ask --search "largura mínima escada"   # retrieval only, no LLM
"""

from __future__ import annotations

import argparse
import sys

from .graph import build_graph, initial_state, text_of
from .retrieval import search, vector_index_ready


def _print_search(query: str) -> None:
    mode = "hybrid (BM25 + vectors)" if vector_index_ready() else "BM25 only (no vector index yet)"
    print(f"retrieval mode: {mode}\n")
    for h in search(query, k=6):
        c = h.chunk
        print(f"{h.score:.4f}  bm25#{h.bm25_rank or '-':<3} vec#{h.vector_rank or '-':<3} [{c['id']}] {c['heading'][:70]}")
        print("        " + c["text"][:160].replace("\n", " "))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true", help="print each tool call")
    parser.add_argument("--search", action="store_true", help="only run retrieval")
    args = parser.parse_args()

    if args.search:
        _print_search(args.question)
        return

    graph = build_graph()
    final: dict = {}
    for mode, payload in graph.stream(initial_state(args.question), stream_mode=["updates", "values"]):
        if mode == "values":
            final = payload
            continue
        if not args.trace:
            continue
        for node, update in payload.items():
            for m in (update or {}).get("messages", []):
                for call in getattr(m, "tool_calls", None) or []:
                    print(f"-> {call['name']}({call['args']})")
                if node == "check_citations":
                    print("-> citation check failed, asking the agent to fix it")

    print("\n" + text_of(final["messages"][-1]).strip() + "\n")
    for c in final.get("citations", []):
        print(f"[{c['id']}] {c['title']}, {c['locator']}\n    {c['url']}")


if __name__ == "__main__":
    main()
