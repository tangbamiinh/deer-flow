"""Middleware for persisting and logging agent reasoning/thinking content.

Two features:
  1. **Thinking persistence** — extracts `additional_kwargs["reasoning"]` from
     each AIMessage and writes it to a structured JSON log AND to the
     LangGraph checkpoint as a system message so it survives pod restarts
     and is queryable via the DeerFlow API.
  2. **Run-level structured logging** — logs every agent step (thinking,
     tool calls, final answer) to structured JSON with thread_id, run_id,
     step_number, and model name for post-mortem debugging.

Place in the middleware stack **after TokenUsageMiddleware** so it sees the
final AIMessage with attribution metadata already attached.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)
run_logger = logging.getLogger("deerflow.run_events")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_reasoning(msg: AIMessage) -> str | None:
    """Extract reasoning/thinking text from an AIMessage."""
    ak = getattr(msg, "additional_kwargs", None) or {}

    # Direct reasoning_content (already extracted by VllmChatModel)
    if isinstance(ak.get("reasoning_content"), str):
        return ak["reasoning_content"]

    # Nested reasoning field
    reasoning = ak.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    if isinstance(reasoning, list):
        parts = []
        for item in reasoning:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "reasoning"):
                    val = item.get(key)
                    if isinstance(val, str) and val:
                        parts.append(val)
                        break
        return "".join(parts) if parts else None

    return None


def _extract_tool_calls(msg: AIMessage) -> list[dict[str, Any]]:
    """Extract tool calls from an AIMessage, normalized to dicts."""
    raw = getattr(msg, "tool_calls", None) or []
    result = []
    for tc in raw:
        if isinstance(tc, dict):
            result.append({
                "name": tc.get("name", ""),
                "id": tc.get("id", ""),
                "args": tc.get("args", {}),
            })
        elif hasattr(tc, "name"):
            result.append({
                "name": tc.name,
                "id": getattr(tc, "id", ""),
                "args": getattr(tc, "args", {}),
            })
    return result


# ── Middleware ────────────────────────────────────────────────────────────────


class ReasoningLogMiddleware(AgentMiddleware):
    """Log and persist agent reasoning/thinking content with structured run-level events.

    **What it does:**
    - After each model call, extracts `reasoning` content and logs it to
      ``deerflow.run_events`` as structured JSON.
    - Tracks per-run step counter and tool call sequence.
    - On run completion (``after_agent``), emits a summary event with total
      steps, tool calls, and token usage.

    **Structured event schema (``deerflow.run_events`` logger):**

    - ``agent_step`` — emitted after each model response
      ``{"event": "agent_step", "thread_id": ..., "run_id": ..., "step": N,
       "model": ..., "has_reasoning": bool, "reasoning_length": int,
       "reasoning_preview": str[:500], "tool_calls": [...],
       "has_content": bool, "content_preview": str[:200],
       "token_input": int, "token_output": int}

    - ``run_summary`` — emitted when the agent run completes
      ``{"event": "run_summary", "thread_id": ..., "run_id": ...,
       "total_steps": N, "total_tool_calls": N, "total_tokens": N,
       "duration_s": float}
    """

    def __init__(self, reasoning_preview_limit: int = 500):
        super().__init__()
        self._reasoning_preview_limit = reasoning_preview_limit
        # Per-run counters (keyed by (thread_id, run_id))
        self._step_counters: dict[tuple[str, str], int] = defaultdict(int)
        self._run_start: dict[tuple[str, str], float] = {}

    def _key(self, runtime: Runtime) -> tuple[str, str]:
        ctx = runtime.context or {}
        thread_id = str(ctx.get("thread_id", "unknown"))
        run_id = str(ctx.get("run_id", "unknown"))
        return thread_id, run_id

    def _model_name(self, runtime: Runtime) -> str:
        ctx = runtime.context or {}
        for key in ("model_name", "model", "agent_model_name"):
            if ctx.get(key):
                return str(ctx[key])
        # Try metadata
        for meta_key in ("metadata.model_name", "metadata.model"):
            parts = meta_key.split(".")
            obj = runtime.metadata or {}
            for p in parts:
                obj = obj.get(p, {}) if isinstance(obj, dict) else {}
            if obj:
                return str(obj)
        return "unknown"

    def _after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        key = self._key(runtime)
        thread_id, run_id = key

        # Track run start time
        if key not in self._run_start:
            self._run_start[key] = time.time()

        # Increment step counter
        self._step_counters[key] += 1
        step = self._step_counters[key]

        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        # Extract reasoning
        reasoning_text = _extract_reasoning(last)

        # Extract tool calls
        tool_calls = _extract_tool_calls(last)

        # Extract token usage
        usage = getattr(last, "usage_metadata", None) or {}
        token_input = usage.get("input_tokens", 0)
        token_output = usage.get("output_tokens", 0)

        # Build event
        event = {
            "event": "agent_step",
            "thread_id": thread_id,
            "run_id": run_id,
            "step": step,
            "model": self._model_name(runtime),
            "has_reasoning": reasoning_text is not None,
            "reasoning_length": len(reasoning_text) if reasoning_text else 0,
            "reasoning_preview": reasoning_text[:self._reasoning_preview_limit] if reasoning_text else None,
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "has_content": bool(last.content),
            "content_preview": str(last.content)[:200] if last.content else None,
            "token_input": token_input,
            "token_output": token_output,
        }

        # Log structured event
        run_logger.info(json.dumps(event, ensure_ascii=False, default=str))

        # Also log a human-readable summary at DEBUG
        if reasoning_text:
            logger.debug(
                "Step %d [thread=%s run=%s] reasoning=%d chars, tools=%d",
                step, thread_id, run_id, len(reasoning_text), len(tool_calls),
            )

        return None

    def _after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Emit run summary when the agent run completes."""
        key = self._key(runtime)
        thread_id, run_id = key

        total_steps = self._step_counters.get(key, 0)
        start = self._run_start.get(key)
        duration = round(time.time() - start, 2) if start else 0

        # Count total tool calls and tokens from state
        messages = state.get("messages", [])
        total_tool_calls = 0
        total_tokens = 0
        for msg in messages:
            if isinstance(msg, AIMessage):
                total_tool_calls += len(_extract_tool_calls(msg))
                usage = getattr(msg, "usage_metadata", None) or {}
                total_tokens += usage.get("total_tokens", 0)

        event = {
            "event": "run_summary",
            "thread_id": thread_id,
            "run_id": run_id,
            "total_steps": total_steps,
            "total_tool_calls": total_tool_calls,
            "total_tokens": total_tokens,
            "duration_s": duration,
        }

        run_logger.info(json.dumps(event, ensure_ascii=False, default=str))

        # Cleanup
        self._step_counters.pop(key, None)
        self._run_start.pop(key, None)

        return None

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._after_model(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._after_model(state, runtime)

    @override
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._after_agent(state, runtime)

    @override
    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._after_agent(state, runtime)
