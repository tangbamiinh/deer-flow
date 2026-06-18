"""JWT interceptor for DeerFlow MCP tools.

Reads user_jwt from LangGraph's runtime context (via get_config) and
injects it as an Authorization header on MCP tool calls (SSE/HTTP transports).

DeerFlow calls jwt_interceptor() to get the actual interceptor callable.
"""
import logging
from contextlib import suppress
logger = logging.getLogger(__name__)


def jwt_interceptor():
    """Builder: return the async interceptor function."""

    async def _interceptor(request, handler):
        """Inject user JWT from LangGraph runtime context into MCP tool call headers."""
        user_jwt = None

        # Try request.runtime first (InjectedToolArg path)
        if request.runtime is not None:
            runtime_ctx = getattr(request.runtime, "context", None)
            if isinstance(runtime_ctx, dict):
                user_jwt = runtime_ctx.get("user_jwt")

        # Fallback: read from LangGraph's current config (works when tool is called
        # directly without InjectedToolArg injection)
        if not user_jwt:
            try:
                from langgraph.config import get_config
                cfg = get_config()
                configurable = cfg.get("configurable", {}) if cfg else {}
                user_jwt = configurable.get("user_jwt")
            except Exception:
                pass

        # Fallback: read from LangChain's get_config
        if not user_jwt:
            try:
                from langchain_core.callbacks import get_trace_context
                # Try langchain_core run tree / callback context
                pass
            except Exception:
                pass

        logger.info("jwt_interceptor: tool=%s, user_jwt=%s, existing_headers=%s",
                     getattr(request, 'name', '?'), bool(user_jwt), request.headers)

        if user_jwt:
            existing_headers = dict(request.headers) if request.headers else {}
            existing_headers["Authorization"] = f"Bearer {user_jwt}"
            request = request.override(headers=existing_headers)
            logger.info("jwt_interceptor: injected JWT successfully")

        return await handler(request)

    return _interceptor
