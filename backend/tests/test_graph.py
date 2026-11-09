"""Graph wiring tests with a scripted fake LLM (no API key needed)."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from coe_agent import retrieval
from coe_agent.graph import build_graph, initial_state, text_of


class ScriptedLLM(GenericFakeChatModel):
    disable_streaming: bool = True  # fake model cannot stream tool calls

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture(autouse=True)
def bm25_only(monkeypatch):
    monkeypatch.setattr(retrieval, "_vectors", lambda: None)


def search_call(query: str, call_id: str = "c1") -> AIMessage:
    return AIMessage("", tool_calls=[{"id": call_id, "name": "search_code", "args": {"query": query, "scope": "anexos"}}])


def run(script: list[AIMessage]) -> dict:
    graph = build_graph(llm=ScriptedLLM(messages=iter(script)))
    return graph.invoke(initial_state("Qual o pé-direito mínimo de um dormitório?"))


def test_valid_citation_passes_check():
    final = run(
        [
            search_call("pé-direito dimensionamento mínimo compartimentos"),
            AIMessage("2,50 m para compartimentos de repouso [D57776-anexos:p9]."),
        ]
    )
    assert [c["id"] for c in final["citations"]] == ["D57776-anexos:p9"]
    assert final["fix_attempts"] == 0


def test_invented_citation_triggers_one_rewrite():
    final = run(
        [
            search_call("pé-direito dimensionamento mínimo compartimentos"),
            AIMessage("2,70 m [L16642:art-999]."),  # id never returned by a tool
            AIMessage("2,50 m [D57776-anexos:p9]."),
        ]
    )
    feedback = [m for m in final["messages"] if isinstance(m, HumanMessage) and "citation check" in m.content]
    assert len(feedback) == 1 and "L16642:art-999" in feedback[0].content
    assert text_of(final["messages"][-1]) == "2,50 m [D57776-anexos:p9]."
    assert [c["id"] for c in final["citations"]] == ["D57776-anexos:p9"]


def test_gives_up_after_one_retry():
    final = run([AIMessage("Não sei."), AIMessage("Ainda não sei.")])
    assert final["fix_attempts"] == 1
    assert final["citations"] == []


def test_server_stream_resets_unverified_answer():
    import asyncio
    import json

    from coe_agent import server

    server._graph = build_graph(
        llm=ScriptedLLM(
            messages=iter(
                [
                    search_call("pé-direito dimensionamento mínimo compartimentos"),
                    AIMessage("2,70 m [L16642:art-999]."),
                    AIMessage("2,50 m [D57776-anexos:p9]."),
                ]
            )
        )
    )

    async def collect():
        msgs = [{"role": "user", "parts": [{"type": "text", "text": "pé-direito?"}]}]
        return [line async for line in server.stream_agent(msgs)]

    try:
        lines = asyncio.run(collect())
    finally:
        server._graph = None
    assert lines[-1] == "data: [DONE]\n\n"
    parts = [json.loads(line[6:]) for line in lines[:-1]]
    types = [p["type"] for p in parts]
    assert types[0] == "start" and types[-1] == "finish"
    assert "error" not in types
    assert types.index("tool-input-available") < types.index("tool-output-available") < types.index("reset-step")
    assert types.count("start-step") == types.count("finish-step")
    text_after_reset = "".join(p.get("delta", "") for p in parts[types.index("reset-step") :])
    assert text_after_reset == "2,50 m [D57776-anexos:p9]."
    citations = next(p for p in parts if p["type"] == "data-citations")["data"]["citations"]
    assert citations[0]["id"] == "D57776-anexos:p9"
