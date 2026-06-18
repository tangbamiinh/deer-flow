"""
ZhiHeng tools for DeerFlow — direct Python registration.

Mounted as a standalone module in the DeerFlow container. No MCP bridge needed.

Each tool is exported as a LangChain StructuredTool instance so DeerFlow's
resolve_variable(cfg.use, BaseTool) accepts it.

Imported in config.yaml via:
  tools:
  - name: lightrag_query
    use: deerflow_tools:lightrag_query
"""
import json
import os
import re

from langchain_core.tools import StructuredTool

import httpx

# ── Configuration ──────────────────────────────────────
BACKEND_URL = os.getenv("DEERFLOW_BACKEND_URL", "http://zhiheng-backend:8000")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "zhiheng-deerflow-internal-2026")
PPTX_SERVICE_URL = os.getenv("PPTX_SERVICE_URL", "http://pptx:8004")


# ── Helpers ────────────────────────────────────────────

async def _http_get(path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{BACKEND_URL}{path}",
            params=params,
            headers={"Authorization": f"Bearer {INTERNAL_API_KEY}"},
        )
        resp.raise_for_status()
        return resp.json()


async def _http_post(path: str, json_body: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{BACKEND_URL}{path}",
            json=json_body,
            headers={"Authorization": f"Bearer {INTERNAL_API_KEY}"},
        )
        resp.raise_for_status()
        return resp.json()


# ── RAG Tools ─────────────────────────────────────────

async def _lightrag_query_impl(query: str, rag_engine: str = "kohakurag", top_k: int = 10) -> str:
    """Query the shared LightRAG knowledge base for course materials and textbooks."""
    try:
        data = await _http_post(
            "/internal/lightrag/query",
            {"query": query, "rag_engine": rag_engine, "top_k": top_k},
        )
        results = data.get("results", [])
        if not results:
            return "未找到相关知识库结果。"
        formatted = "\n\n".join(
            f"[{i+1}] {r.get('content', '')[:500]}"
            for i, r in enumerate(results[:top_k])
        )
        return f"知识库查询结果:\n\n{formatted}"
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"


async def _kohakurag_search_impl(query: str, top_k: int = 10) -> str:
    """Search the shared KohakuRAG textbook index."""
    try:
        data = await _http_post(
            "/internal/kohakurag/search",
            {"query": query, "top_k": top_k},
        )
        results = data.get("results", [])
        if not results:
            return "未找到相关教材结果。"
        formatted = "\n".join(
            f"  - {c.get('file_path', 'N/A')}: {c.get('content', '')[:200]}"
            for c in results[:top_k]
        )
        return f"教材搜索结果:\n{formatted}"
    except Exception as e:
        return f"Error searching textbooks: {str(e)}"


async def _kohakurag_qa_impl(query: str, top_k: int = 10) -> str:
    """Ask a question to the KohakuRAG Q&A system — gets LLM-generated answers with sources."""
    try:
        data = await _http_post(
            "/internal/kohakurag/qa",
            {"query": query, "top_k": top_k},
        )
        answer = data.get("answer", "未生成回答。")
        chunks = data.get("chunks", [])
        if chunks:
            sources_text = "\n".join(
                f"  - {c.get('file_path', 'N/A')}: {c.get('content', '')[:200]}"
                for c in chunks[:5]
            )
            return f"{answer}\n\n参考来源:\n{sources_text}"
        return answer
    except Exception as e:
        return f"Error with Q&A: {str(e)}"


# ── Lesson Tools ──────────────────────────────────────

async def _list_lessons_impl(user_id: str) -> str:
    """List all lessons for a teacher."""
    try:
        lessons = await _http_get("/internal/lessons", {"user_id": user_id})
        if not lessons:
            return "未找到教案。"
        lines = [
            f"- [{l.get('topic', 'N/A')}] (ID: {l.get('id', 'N/A')[:8]})"
            for l in lessons
        ]
        return "教案列表:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error listing lessons: {str(e)}"


async def _get_lesson_impl(lesson_id: str, user_id: str) -> str:
    """Get a specific lesson by ID."""
    try:
        lesson = await _http_get(f"/internal/lessons/{lesson_id}", {"user_id": user_id})
        return (
            f"教案: {lesson.get('topic', 'N/A')}\n"
            f"状态: {lesson.get('status', 'N/A')}\n"
            f"内容: {lesson.get('content', '')[:2000]}"
        )
    except Exception as e:
        return f"教案未找到或无权限: {str(e)}"


async def _create_lesson_impl(topic: str, content: str, user_id: str) -> str:
    """Create a new lesson for a teacher."""
    try:
        lesson = await _http_post(
            "/internal/lessons",
            {"topic": topic, "content": content, "user_id": user_id},
        )
        return f"教案已创建 (ID: {lesson.get('id', 'unknown')[:8]})"
    except Exception as e:
        return f"Failed to create lesson: {str(e)}"


# ── Presentation Tools ────────────────────────────────

async def _list_presentations_impl(user_id: str) -> str:
    """List all presentations for a teacher."""
    try:
        presentations = await _http_get("/internal/presentations", {"user_id": user_id})
        if not presentations:
            return "未找到课件。"
        lines = [
            f"- [{p.get('title', p.get('topic', 'N/A'))}] (ID: {p.get('id', 'N/A')[:8]}, {p.get('status', 'pending')})"
            for p in presentations
        ]
        return "课件列表:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error listing presentations: {str(e)}"


async def _create_presentation_impl(topic: str, content: str, user_id: str) -> str:
    """Create a new presentation record for a teacher."""
    try:
        pres = await _http_post(
            "/internal/presentations",
            {"topic": topic, "content": content, "user_id": user_id},
        )
        return f"课件已创建 (ID: {pres.get('id', 'unknown')[:8]})"
    except Exception as e:
        return f"Failed to create presentation: {str(e)}"


# ── Web Tools ──────────────────────────────────────────

async def _web_fetch_impl(url: str, max_length: int = 5000) -> str:
    """Fetch and extract text content from a URL. Handles both static and JavaScript-rendered pages via Jina Reader."""
    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(jina_url, headers={"Accept": "text/plain"})
            if resp.status_code == 200:
                text = resp.text.strip()
                if text and len(text) > 100:
                    return f"Content from {url}:\n\n{text[:max_length]}"
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; DeerFlow/1.0)"})
            resp.raise_for_status()
            html = resp.text

            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                return f"Fetched {url} but no text content found. The page may be JavaScript-rendered."
            return f"Content from {url}:\n\n{text[:max_length]}"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


# ── PPT Generation Tools (PPTX service) ────────────────

async def _generate_ppt_impl(manifest_json: str) -> str:
    """Submit a PPTX generation request. Returns task_id immediately. Use check_ppt_status to poll for completion.
    
    Args:
        manifest_json: JSON string of the slide manifest with title, theme, and slides array.
    """
    try:
        manifest = json.loads(manifest_json) if isinstance(manifest_json, str) else manifest_json
        if not manifest.get("slides"):
            return "Error: manifest must contain a non-empty slides array."

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{PPTX_SERVICE_URL}/generate/task",
                json={"manifest": manifest},
            )
            resp.raise_for_status()
            result = resp.json()

        return json.dumps({
            "task_id": result["task_id"],
            "status": result["status"],
            "message": result["message"],
            "preview_url": result.get("preview_url", ""),
            "next_step": "Use check_ppt_status with this task_id to check progress.",
        })
    except Exception as e:
        return f"提交 PPT 生成请求时出错: {str(e)}"


async def _check_ppt_status_impl(task_id: str) -> str:
    """Check the status of a PPTX generation task."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{PPTX_SERVICE_URL}/generate/task/{task_id}/status"
            )
            resp.raise_for_status()
            result = resp.json()

        status = result.get("status", "unknown")
        if status == "complete":
            return json.dumps({
                "status": "complete",
                "message": result["message"],
                "file_key": result.get("file_key", ""),
                "file_size": result.get("file_size", 0),
                "preview_url": result.get("preview_url", ""),
            })
        elif status in ("error", "failed"):
            return json.dumps({
                "status": status,
                "error": result.get("error", "未知错误"),
            })
        else:
            return json.dumps({
                "status": status,
                "message": result.get("message", "Generating..."),
                "events_count": result.get("events_count", 0),
            })
    except Exception as e:
        return f"查询 PPT 生成状态时出错: {str(e)}"


# ── StructuredTool instances (what DeerFlow expects) ──

lightrag_query = StructuredTool.from_function(
    coroutine=_lightrag_query_impl,
    name="lightrag_query",
    description="Query the shared LightRAG knowledge base for course materials and textbooks.",
)

kohakurag_search = StructuredTool.from_function(
    coroutine=_kohakurag_search_impl,
    name="kohakurag_search",
    description="Search the shared KohakuRAG textbook index.",
)

kohakurag_qa = StructuredTool.from_function(
    coroutine=_kohakurag_qa_impl,
    name="kohakurag_qa",
    description="Ask a question to the KohakuRAG Q&A system — gets LLM-generated answers with sources.",
)

list_lessons = StructuredTool.from_function(
    coroutine=_list_lessons_impl,
    name="list_lessons",
    description="List all lessons for a teacher.",
)

get_lesson = StructuredTool.from_function(
    coroutine=_get_lesson_impl,
    name="get_lesson",
    description="Get a specific lesson by ID.",
)

create_lesson = StructuredTool.from_function(
    coroutine=_create_lesson_impl,
    name="create_lesson",
    description="Create a new lesson for a teacher.",
)

list_presentations = StructuredTool.from_function(
    coroutine=_list_presentations_impl,
    name="list_presentations",
    description="List all presentations for a teacher.",
)

create_presentation = StructuredTool.from_function(
    coroutine=_create_presentation_impl,
    name="create_presentation",
    description="Create a new presentation record for a teacher.",
)

web_fetch = StructuredTool.from_function(
    coroutine=_web_fetch_impl,
    name="web_fetch",
    description="Fetch and extract text content from a URL. Handles both static and JavaScript-rendered pages via Jina Reader.",
)

generate_ppt = StructuredTool.from_function(
    coroutine=_generate_ppt_impl,
    name="generate_ppt",
    description="Submit a PPTX generation request. Accepts a JSON manifest with title, theme, and slides array. Returns task_id immediately. Use check_ppt_status to poll for completion.",
)

check_ppt_status = StructuredTool.from_function(
    coroutine=_check_ppt_status_impl,
    name="check_ppt_status",
    description="Check the status of a PPTX generation task by task_id.",
)
