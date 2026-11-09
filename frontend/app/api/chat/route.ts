// Proxies useChat requests to the Python LangGraph agent, which already speaks the
// AI SDK UI message stream protocol. Keeping the proxy here means the browser only ever
// talks to Next.js, and the backend URL stays server-side.

const BACKEND_URL = process.env.COE_API_URL ?? "http://127.0.0.1:8000";

export const maxDuration = 120;

export async function POST(request: Request) {
  const upstream = await fetch(`${BACKEND_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: await request.text(),
    signal: request.signal,
  }).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    return Response.json(
      { error: `Agent backend not reachable at ${BACKEND_URL}: ${message}` },
      { status: 502 },
    );
  });

  if (!upstream.ok || !upstream.body) return upstream;

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "x-vercel-ai-ui-message-stream": "v1",
    },
  });
}
