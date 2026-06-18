"""Middleware for intercepting present_manifest calls and presenting manifests to the user."""

import json
import logging
from collections.abc import Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class PresentManifestMiddlewareState(AgentState):
    """Compatible with the ThreadState schema."""
    pass


class PresentManifestMiddleware(AgentMiddleware[PresentManifestMiddlewareState]):
    """Intercept present_manifest tool calls and interrupt execution.

    When the model calls present_manifest, this middleware:
    1. Intercepts the tool call before execution
    2. Extracts the manifest JSON and slide count
    3. Creates a ToolMessage with the manifest data
    4. Returns Command(goto=END) to interrupt execution
    5. Frontend detects the ToolMessage and shows the ManifestEditor overlay
    6. Waits for user response before continuing
    """

    state_schema = PresentManifestMiddlewareState

    def _handle_present_manifest(self, request: ToolCallRequest) -> Command:
        """Handle present_manifest request and return command to interrupt execution."""
        args = request.tool_call.get("args", {})

        # Extract manifest - may be a dict or JSON string
        manifest = args.get("manifest", {})
        if isinstance(manifest, str):
            try:
                manifest = json.loads(manifest)
            except (json.JSONDecodeError, TypeError):
                manifest = {}

        slide_count = args.get("slide_count", 0)
        if not slide_count:
            slide_count = len(manifest.get("slides", []))

        logger.info("Intercepted present_manifest request with %d slides", slide_count)

        # Build ToolMessage content - structured JSON for frontend detection
        message_content = json.dumps({
            "type": "present_manifest",
            "manifest": manifest,
            "slide_count": slide_count,
        })

        tool_call_id = request.tool_call.get("id", "")
        msg_id = f"manifest:{tool_call_id}" if tool_call_id else "manifest:default"

        tool_message = ToolMessage(
            id=msg_id,
            content=message_content,
            tool_call_id=tool_call_id,
            name="present_manifest",
        )

        return Command(
            update={"messages": [tool_message]},
            goto=END,
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if request.tool_call.get("name") != "present_manifest":
            return handler(request)
        return self._handle_present_manifest(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if request.tool_call.get("name") != "present_manifest":
            return await handler(request)
        return self._handle_present_manifest(request)
