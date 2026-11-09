"""LangGraph agent: plan -> call tools -> answer -> verify citations.

    START -> agent --(tool calls)--> tools -> agent
                   --(answer)------> check_citations --(ok / retried once)--> END
                                                     --(bad citations)------> agent

`check_citations` is the guardrail: every [id] in the answer must be a passage that a tool
actually returned in this conversation. If not, the agent gets one chance to fix it.
"""

from __future__ import annotations

import re
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import CHAT_MODEL, SOURCES_BY_KEY
from .tools import TOOLS

# Citations look like "[L16642:art-31]". Models sometimes put an inciso or a second id in the
# same brackets ("[L16642:art-15, II]", "[L16642:art-12-1, D57776:art-71]"), so every id inside
# any bracket group counts, and the rest of the group is ignored.
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
ID_RE = re.compile(r"[A-Za-z0-9-]+:(?:art-[0-9A-Z]+(?:-[0-9A-Z]+)*|p[0-9]+)")
MAX_FIX_ATTEMPTS = 1


def find_citations(text: str) -> list[str]:
    """Cited ids in order of first appearance."""
    ids = [i for group in BRACKET_RE.findall(text) for i in ID_RE.findall(group)]
    return list(dict.fromkeys(ids))

SYSTEM_PROMPT = """You are an assistant for architects and engineers working with the \
São Paulo Building Code (Código de Obras e Edificações - COE): Lei nº 16.642/2017, its \
regulation Decreto nº 57.776/2017, and the technical annexes of both.

How to work:
- Always search before answering. Never answer from memory: the code changes and your \
memory may be wrong or from another city.
- Search with Portuguese legal terms, even if the question is in English. Try 2-3 different \
phrasings if the first search misses. Use read_article when a passage cites another article, \
and read_annex_page to see a whole technical table.
- Dimensions and numbers usually live in the annexes (scope="anexos"); procedures and \
responsibilities live in the articles. Definitions of terms (reforma, reparo, recuo, \
edificação...) are in art. 3 of the Lei: search for the term itself to find them.

How to answer:
- Write the whole answer in the language of the question. For an English question, answer \
in English and keep Portuguese legal terms in italics.
- Be short and practical, like a note to a colleague on site: lead with the direct answer \
(the number, the rule), then conditions and exceptions.
- Every sentence that states a rule, a number or a definition must end with a citation: the \
passage id in square brackets, exactly as given by the tools, e.g. [L16642:art-31] or \
[D57776-anexos:p9]. If you cannot cite a statement, leave it out. Only cite ids you received \
from a tool.
- One id per bracket: [L16642:art-3-4] [D57776:art-71], never [L16642:art-3-4, D57776:art-71]. \
Write incisos and paragraphs in the text ("art. 3, XXX [L16642:art-3-4]"). Do not add a \
reference list at the end: the interface builds it from your citations.
- If the passages do not answer the question, say so plainly and suggest where to look \
(e.g. LPUOS zoning law, NBR 9050, fire department rules). Do not guess.
- This is not legal advice; mention that only when the answer depends on interpretation."""


def _merge_sources(left: dict[str, dict], right: dict[str, dict]) -> dict[str, dict]:
    return {**(left or {}), **(right or {})}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    sources: Annotated[dict[str, dict], _merge_sources]  # every passage the tools returned
    citations: list[dict]  # passages cited in the final answer, in order
    fix_attempts: int


def _collect_sources(messages: list[AnyMessage]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(m.artifact, list):
            for chunk in m.artifact:
                found[chunk["id"]] = chunk
    return found


def text_of(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def build_graph(model: str = CHAT_MODEL, llm=None):
    """`llm` lets tests inject a fake chat model; by default any LangChain provider string
    works, e.g. "google_genai:gemini-3.5-flash" or "openai:gpt-4.1-mini"."""
    llm = (llm or init_chat_model(model, temperature=0, max_retries=2)).bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def agent(state: State) -> dict:
        response = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    def tools(state: State) -> dict:
        result = tool_node.invoke(state)
        return {"messages": result["messages"], "sources": _collect_sources(result["messages"])}

    def check_citations(state: State) -> dict:
        answer = text_of(state["messages"][-1])
        known = state.get("sources") or {}
        cited = find_citations(answer)
        unknown = [c for c in cited if c not in known]
        attempts = state.get("fix_attempts", 0)

        if (unknown or not cited) and attempts < MAX_FIX_ATTEMPTS:
            problem = (
                f"these ids were not returned by any tool: {', '.join(unknown)}"
                if unknown
                else "the answer has no citations"
            )
            return {
                "fix_attempts": attempts + 1,
                "messages": [
                    HumanMessage(
                        f"[citation check] Your answer cannot be shown yet: {problem}. "
                        "Search again if needed, then rewrite the full answer citing only "
                        "passage ids returned by the tools. If the documents do not cover "
                        "the question, say so without citations."
                    )
                ],
            }
        citations = []
        for cid in cited:
            if cid in known:
                c = known[cid]
                citations.append(
                    {
                        "id": cid,
                        "title": SOURCES_BY_KEY[c["source"]].title,
                        "locator": c["locator"],
                        "url": c["url"],
                        "snippet": c["text"][:280],
                    }
                )
        return {"citations": citations}

    def route_after_agent(state: State) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "check_citations"

    def route_after_check(state: State) -> str:
        last = state["messages"][-1]
        return "agent" if isinstance(last, HumanMessage) else END

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("check_citations", check_citations)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "check_citations"])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("check_citations", route_after_check, ["agent", END])
    return graph.compile()


def initial_state(question: str, history: list[AnyMessage] | None = None) -> State:
    return {
        "messages": [*(history or []), HumanMessage(question)],
        "sources": {},
        "citations": [],
        "fix_attempts": 0,
    }
