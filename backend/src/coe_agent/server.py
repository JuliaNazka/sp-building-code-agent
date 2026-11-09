"""FastAPI server that streams the LangGraph agent in the Vercel AI SDK UI message stream
protocol, so the Next.js front end can use `useChat` directly.

    coe-serve            # http://127.0.0.1:8000
    POST /api/chat       # body: { messages: UIMessage[] } (what useChat sends)

Mapping from graph events to stream parts:
- agent tokens            -> text-start / text-delta / text-end
- agent tool calls        -> tool-input-available (dynamic tools)
- tool results            -> tool-output-available (passage ids + locators)
- failed citation check   -> reset-step (drops the unverified answer) + data-status
- verified citations      -> source-url parts + data-citations
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage

from .graph import build_graph, initial_state
from .retrieval import vector_index_ready

app = FastAPI(title="COE agent")
_graph = None


def graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def sse(part: dict) -> str:
    return f"data: {json.dumps(part, ensure_ascii=False)}\n\n"


def _ui_text(message: dict) -> str:
    parts = message.get("parts") or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def to_langchain(messages: list[dict]) -> tuple[list[AnyMessage], str]:
    history: list[AnyMessage] = []
    for m in messages[:-1]:
        text = _ui_text(m)
        if not text:
            continue
        history.append(HumanMessage(text) if m.get("role") == "user" else AIMessage(text))
    return history, _ui_text(messages[-1])


def _chunk_text(chunk: AIMessageChunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def _tool_summary(msg: ToolMessage) -> dict:
    passages = msg.artifact if isinstance(msg.artifact, list) else []
    return {
        "passages": [{"id": c["id"], "locator": c["locator"], "heading": c["heading"]} for c in passages],
        "count": len(passages),
    }


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (
            "The Gemini free-tier quota for this model is used up (requests per minute or per day). "
            "Wait a bit, or set COE_CHAT_MODEL in .env to another model."
        )
    if "API_KEY" in text.upper() or "PERMISSION_DENIED" in text:
        return "Gemini rejected the API key. Check GOOGLE_API_KEY in .env."
    return f"{exc.__class__.__name__}: {text[:300]}"


async def stream_agent(messages: list[dict]) -> AsyncIterator[str]:
    history, question = to_langchain(messages)
    yield sse({"type": "start", "messageId": f"msg_{uuid.uuid4().hex}"})

    text_id: str | None = None
    in_step = False

    def open_step() -> list[dict]:
        nonlocal in_step
        if in_step:
            return []
        in_step = True
        return [{"type": "start-step"}]

    def close_text() -> list[dict]:
        nonlocal text_id
        if text_id is None:
            return []
        part = {"type": "text-end", "id": text_id}
        text_id = None
        return [part]

    def close_step() -> list[dict]:
        nonlocal in_step
        out = close_text()
        if in_step:
            out.append({"type": "finish-step"})
            in_step = False
        return out

    try:
        state = initial_state(question, history)
        async for mode, payload in graph().astream(state, stream_mode=["messages", "updates"]):
            if mode == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessageChunk):
                    continue
                delta = _chunk_text(chunk)
                if not delta:
                    continue
                parts = open_step()
                if text_id is None:
                    text_id = f"txt_{uuid.uuid4().hex[:12]}"
                    parts.append({"type": "text-start", "id": text_id})
                parts.append({"type": "text-delta", "id": text_id, "delta": delta})
                for p in parts:
                    yield sse(p)
                continue

            for node, update in payload.items():
                update = update or {}
                if node == "agent":
                    msg = update["messages"][-1]
                    parts = []
                    if text_id is None and not msg.tool_calls:
                        # the model did not stream tokens (some providers / fakes): send it whole
                        full = _chunk_text(msg)
                        if full:
                            tid = f"txt_{uuid.uuid4().hex[:12]}"
                            parts = open_step() + [
                                {"type": "text-start", "id": tid},
                                {"type": "text-delta", "id": tid, "delta": full},
                                {"type": "text-end", "id": tid},
                            ]
                    parts += close_text()
                    if msg.tool_calls:
                        parts = open_step() + parts
                        for call in msg.tool_calls:
                            parts.append(
                                {
                                    "type": "tool-input-available",
                                    "toolCallId": call["id"],
                                    "toolName": call["name"],
                                    "input": call["args"],
                                    "dynamic": True,
                                }
                            )
                    for p in parts:
                        yield sse(p)
                elif node == "tools":
                    for msg in update.get("messages", []):
                        yield sse(
                            {
                                "type": "tool-output-available",
                                "toolCallId": msg.tool_call_id,
                                "output": _tool_summary(msg),
                                "dynamic": True,
                            }
                        )
                    for p in close_step():
                        yield sse(p)
                elif node == "check_citations":
                    if update.get("messages"):  # citation check failed: drop that answer
                        for p in close_text():
                            yield sse(p)
                        yield sse({"type": "reset-step"})
                        yield sse({"type": "data-status", "data": {"state": "rewriting"}, "transient": True})
                        for p in close_step():
                            yield sse(p)
                        continue
                    citations = update.get("citations", [])
                    for p in close_step():
                        yield sse(p)
                    for c in citations:
                        yield sse(
                            {
                                "type": "source-url",
                                "sourceId": c["id"],
                                "url": c["url"],
                                "title": f"{c['title']}, {c['locator']}",
                            }
                        )
                    yield sse({"type": "data-citations", "data": {"citations": citations}})
        for p in close_step():
            yield sse(p)
        yield sse({"type": "finish"})
    except Exception as exc:  # surface the error in the chat instead of a broken stream
        for p in close_step():
            yield sse(p)
        yield sse({"type": "error", "errorText": friendly_error(exc)})
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    return StreamingResponse(
        stream_agent(body["messages"]),
        media_type="text/event-stream",
        headers={
            "x-vercel-ai-ui-message-stream": "v1",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "vector_index": vector_index_ready()}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.getenv("COE_HOST", "127.0.0.1"), port=int(os.getenv("COE_PORT", "8000")))


if __name__ == "__main__":
    main()
