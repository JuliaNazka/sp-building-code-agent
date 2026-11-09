"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Citation = { id: string; title: string; locator: string; url: string; snippet: string };
type ToolOutput = { count: number; passages: { id: string; locator: string; heading: string }[] };

// Same rule as the backend check: every id inside any [...] group is a citation.
const BRACKET_RE = /\[([^[\]]+)\]/g;
const ID_RE = /[A-Za-z0-9-]+:(?:art-[0-9A-Z]+(?:-[0-9A-Z]+)*|p[0-9]+)/g;

const EXAMPLES = [
  "Qual o pé-direito mínimo e a área mínima de um dormitório?",
  "Quantas vagas para idosos um estacionamento coletivo precisa?",
  "What is a 'reforma sem acréscimo de área' and does it need a permit?",
  "Qual a multa por executar obra sem alvará?",
];

const TOOL_LABELS: Record<string, string> = {
  search_code: "Searching",
  read_article: "Reading article",
  read_annex_page: "Reading annex page",
};

function toolInputLabel(name: string, input: Record<string, unknown> | undefined): string {
  if (!input) return "";
  if (name === "search_code") return `“${input.query}”${input.scope && input.scope !== "all" ? ` in ${input.scope}` : ""}`;
  if (name === "read_article") return `art. ${input.article} (${input.law})`;
  if (name === "read_annex_page") return `p. ${input.page} (${input.annex} annex)`;
  return JSON.stringify(input);
}

function citationsOf(message: UIMessage): Citation[] {
  for (const part of message.parts) {
    if (part.type === "data-citations") return (part.data as { citations: Citation[] }).citations;
  }
  return [];
}

function AnswerText({ text, citations }: { text: string; citations: Citation[] }) {
  const order: string[] = [];
  const withLinks = text.replace(BRACKET_RE, (group: string, inner: string) => {
    const ids = inner.match(ID_RE);
    if (!ids) return group;
    return ids
      .map((id) => {
        if (!order.includes(id)) order.push(id);
        return ` [${order.indexOf(id) + 1}](#cite-${id})`;
      })
      .join("");
  });
  const byId = new Map(citations.map((c) => [c.id, c]));

  return (
    <div className="prose-answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith("#cite-")) {
              const c = byId.get(href.slice(6));
              return (
                <a
                  href={c?.url ?? undefined}
                  target="_blank"
                  rel="noreferrer"
                  title={c ? `${c.title}, ${c.locator}` : href.slice(6)}
                  className="cite"
                >
                  {children}
                </a>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer" className="underline">
                {children}
              </a>
            );
          },
        }}
      >
        {withLinks}
      </ReactMarkdown>
    </div>
  );
}

function AssistantMessage({ message }: { message: UIMessage }) {
  const citations = citationsOf(message);
  const text = message.parts
    .filter((p) => p.type === "text")
    .map((p) => p.text)
    .join("");
  const steps = message.parts.filter((p) => p.type === "dynamic-tool");

  return (
    <div className="flex flex-col gap-3">
      {steps.length > 0 && (
        <ol className="flex flex-col gap-1 text-xs text-[var(--muted)]">
          {steps.map((p) => {
            const out = p.state === "output-available" ? (p.output as ToolOutput) : undefined;
            return (
              <li key={p.toolCallId} className="flex gap-2">
                <span aria-hidden>{out ? "✓" : "…"}</span>
                <span>
                  {TOOL_LABELS[p.toolName] ?? p.toolName} {toolInputLabel(p.toolName, p.input as Record<string, unknown>)}
                  {out && ` → ${out.count} passage${out.count === 1 ? "" : "s"}`}
                </span>
              </li>
            );
          })}
        </ol>
      )}
      {text && <AnswerText text={text} citations={citations} />}
      {citations.length > 0 && (
        <ol className="sources">
          {citations.map((c, i) => (
            <li key={c.id}>
              <span className="cite">{i + 1}</span>
              <a href={c.url} target="_blank" rel="noreferrer">
                <strong>{c.title}</strong>, {c.locator}
              </a>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export default function Chat() {
  const [input, setInput] = useState("");
  const [rewriting, setRewriting] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const { messages, sendMessage, status, error, stop } = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
    onData: (part) => {
      if (part.type === "data-status") setRewriting(true);
    },
    onFinish: () => setRewriting(false),
  });
  const busy = status === "submitted" || status === "streaming";

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function ask(text: string) {
    if (!text.trim() || busy) return;
    setRewriting(false);
    sendMessage({ text });
    setInput("");
  }

  return (
    <div className="mx-auto flex min-h-dvh w-full max-w-3xl flex-col px-4">
      <header className="border-b border-[var(--line)] py-5">
        <h1 className="text-lg font-semibold">COE Agent · São Paulo Building Code</h1>
        <p className="text-sm text-[var(--muted)]">
          Answers from Lei 16.642/2017, Decreto 57.776/2017 and their annexes, with a citation for every claim.
          Not legal advice.
        </p>
      </header>

      <main className="flex flex-1 flex-col gap-6 py-6">
        {messages.length === 0 && (
          <div className="grid gap-2 sm:grid-cols-2">
            {EXAMPLES.map((q) => (
              <button key={q} onClick={() => ask(q)} className="example">
                {q}
              </button>
            ))}
          </div>
        )}

        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="user-bubble">
              {m.parts.map((p) => (p.type === "text" ? p.text : "")).join("")}
            </div>
          ) : (
            <AssistantMessage key={m.id} message={m} />
          ),
        )}

        {busy && (
          <p className="text-xs text-[var(--muted)]">
            {rewriting ? "Citation check failed, rewriting the answer…" : status === "submitted" ? "Thinking…" : ""}
          </p>
        )}
        {error && <p className="text-sm text-red-600">Error: {error.message}</p>}
        <div ref={bottom} />
      </main>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
        className="sticky bottom-0 flex gap-2 border-t border-[var(--line)] bg-[var(--background)] py-4"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the São Paulo building code (Portuguese or English)…"
          className="flex-1 rounded-lg border border-[var(--line)] bg-transparent px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
        />
        {busy ? (
          <button type="button" onClick={stop} className="btn">
            Stop
          </button>
        ) : (
          <button type="submit" className="btn" disabled={!input.trim()}>
            Ask
          </button>
        )}
      </form>
    </div>
  );
}
