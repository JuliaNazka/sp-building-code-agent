"""Optional Langfuse tracing through the LangChain callback system.

Tracing turns on only when LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL
(or the older LANGFUSE_HOST) are set, e.g. in .env. Without them the agent runs exactly as
before and the langfuse package is never imported.

Each graph run becomes one trace with the LLM calls (prompts, tool calls, token usage),
the tool runs and the citation-check node as nested spans.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

RUN_NAME = "coe-agent"


def tracing_enabled() -> bool:
    keys = os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    return bool(keys and host)


def run_config(**metadata) -> dict:
    """LangGraph run config: Langfuse callback when configured, plus a run name and metadata."""
    config: dict = {"run_name": RUN_NAME, "metadata": metadata}
    if tracing_enabled():
        try:
            from langfuse.langchain import CallbackHandler

            config["callbacks"] = [CallbackHandler()]
        except Exception as exc:  # never break answering because of tracing
            logger.warning("Langfuse tracing disabled (%s)", exc)
    return config


def flush() -> None:
    """Send pending spans before a short-lived process (the CLI) exits."""
    if not tracing_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        logger.warning("Langfuse flush failed (%s)", exc)
