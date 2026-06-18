"""Redis-backed memory storage for DeerFlow.

Implements MemoryStorage ABC — replaces ephemeral file-based memory with
persistent Redis storage. Survives pod restarts, multi-worker safe.

Redis keys:
  deerflow:memory:{user_id}          — per-user memory JSON
  deerflow:memory:default            — global fallback (no user_id)

Config wire (config.yaml):
  memory:
    storage_class: deerflow_redis_memory.RedisMemoryStorage

Requires: redis>=4.0 (installed via uv pip install at deploy time)
"""

import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any

import redis

logger = logging.getLogger(__name__)


def utc_now_iso_z() -> str:
    return datetime.now(UTC).isoformat().removesuffix("+00:00") + "Z"


def create_empty_memory() -> dict[str, Any]:
    return {
        "version": "1.0",
        "lastUpdated": utc_now_iso_z(),
        "user": {
            "workContext": {"summary": "", "updatedAt": ""},
            "personalContext": {"summary": "", "updatedAt": ""},
            "topOfMind": {"summary": "", "updatedAt": ""},
        },
        "history": {
            "recentMonths": {"summary": "", "updatedAt": ""},
            "earlierContext": {"summary": "", "updatedAt": ""},
            "longTermBackground": {"summary": "", "updatedAt": ""},
        },
        "facts": [],
    }


def _memory_key(user_id: str | None, agent_name: str | None) -> str:
    prefix = f"deerflow:memory:{user_id}" if user_id else "deerflow:memory:default"
    suffix = agent_name or "default"
    return f"{prefix}:{suffix}"


class RedisMemoryStorage:
    """Persistent memory storage backed by Redis.

    Thread-safe, multi-worker safe. Uses atomic GET/SET (single key) so
    concurrent writes serialize through Redis — no TOCTOU race.

    Registered as virtual subclass of deerflow MemoryStorage via module-level
    .register() call so get_memory_storage() issubclass() check passes.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        redis_host: str | None = None,
        redis_port: int = 6379,
        redis_db: int = 0,
    ):
        if redis_url:
            self._client: redis.Redis = redis.Redis.from_url(
                redis_url, decode_responses=True
            )
        elif redis_host:
            self._client = redis.Redis(
                host=redis_host, port=redis_port, db=redis_db, decode_responses=True
            )
        else:
            self._client = redis.Redis.from_url(
                "redis://default:zhiheng_redis_2026@zhiheng-redis:6379/0", decode_responses=True
            )

        self._cache: dict[tuple[str | None, str | None], dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

        try:
            self._client.ping()
            logger.info("RedisMemoryStorage connected to %s", redis_url or "default")
        except redis.ConnectionError as e:
            logger.error("RedisMemoryStorage connection failed: %s", e)
            raise

        # Register as virtual subclass of MemoryStorage for issubclass() check.
        try:
            from deerflow.agents.memory.storage import MemoryStorage
            MemoryStorage.register(RedisMemoryStorage)
            logger.info("RedisMemoryStorage registered as MemoryStorage subclass")
        except Exception as e:
            logger.warning("Could not register RedisMemoryStorage: %s", e)

    @staticmethod
    def _cache_key(user_id: str | None, agent_name: str | None) -> tuple[str | None, str | None]:
        return (user_id, agent_name)

    def _load_from_redis(self, key: str) -> dict[str, Any]:
        try:
            data = self._client.get(key)
            if data is None:
                return create_empty_memory()
            memory = json.loads(data)
            if not isinstance(memory, dict) or "facts" not in memory:
                logger.warning("Corrupt memory data for key %s, resetting", key)
                return create_empty_memory()
            return memory
        except (json.JSONDecodeError, redis.RedisError) as e:
            logger.warning("Failed to load memory from Redis %s: %s", key, e)
            return create_empty_memory()

    def load(
        self,
        agent_name: str | None = None,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        rkey = _memory_key(user_id, agent_name)
        ckey = self._cache_key(user_id, agent_name)
        memory_data = self._load_from_redis(rkey)
        with self._cache_lock:
            self._cache[ckey] = memory_data
        return memory_data

    def reload(
        self,
        agent_name: str | None = None,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        rkey = _memory_key(user_id, agent_name)
        ckey = self._cache_key(user_id, agent_name)
        memory_data = self._load_from_redis(rkey)
        with self._cache_lock:
            self._cache[ckey] = memory_data
        return memory_data

    def save(
        self,
        memory_data: dict[str, Any],
        agent_name: str | None = None,
        *,
        user_id: str | None = None,
    ) -> bool:
        rkey = _memory_key(user_id, agent_name)
        ckey = self._cache_key(user_id, agent_name)
        try:
            memory_data = {**memory_data, "lastUpdated": utc_now_iso_z()}
            payload = json.dumps(memory_data, ensure_ascii=False)
            self._client.set(rkey, payload)
            with self._cache_lock:
                self._cache[ckey] = memory_data
            logger.debug("Memory saved to Redis key %s (%d facts)", rkey, len(memory_data.get("facts", [])))
            return True
        except (redis.RedisError, TypeError) as e:
            logger.error("Failed to save memory to Redis %s: %s", rkey, e)
            return False


# Register as virtual subclass of MemoryStorage at module level —
# before get_memory_storage() instantiates the class, so issubclass() passes.
try:
    from deerflow.agents.memory.storage import MemoryStorage
    MemoryStorage.register(RedisMemoryStorage)
    logger.info("RedisMemoryStorage registered as MemoryStorage virtual subclass")
except Exception as e:
    logger.warning("Could not register RedisMemoryStorage: %s", e)


# ── Monkey-patch: fix user_id resolution in DynamicContextMiddleware ──
#
# Bug: DynamicContextMiddleware._build_full_reminder() calls
# _get_memory_context() -> get_effective_user_id() which reads from the
# _current_user ContextVar. When DeerFlow is called via our backend API
# (not through DeerFlow's own auth middleware), this ContextVar is empty,
# so it falls back to "default" and loads the wrong memory key.
#
# Fix: patch before_agent/abefore_agent to set _current_user from
# runtime.context["user_id"] before _inject() runs.
try:
    _patch_logger = logging.getLogger(__name__ + ".middleware_patch")

    from deerflow.runtime.user_context import (
        set_current_user, reset_current_user, get_effective_user_id
    )

    class _FakeUser:
        """Minimal CurrentUser mock — only .id is needed."""
        __slots__ = ("id",)
        def __init__(self, uid: str):
            self.id = uid

    def _resolve_user_id_from_runtime(runtime) -> str | None:
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, dict):
            uid = ctx.get("user_id")
            if uid:
                return str(uid)
        return None

    def _apply_middleware_patch():
        from deerflow.agents.middlewares.dynamic_context_middleware import (
            DynamicContextMiddleware
        )

        _orig_before = DynamicContextMiddleware.before_agent
        _orig_abefore = DynamicContextMiddleware.abefore_agent

        def patched_before(self, state, runtime):
            uid = _resolve_user_id_from_runtime(runtime)
            if uid:
                token = set_current_user(_FakeUser(uid))
                try:
                    return _orig_before(self, state, runtime)
                finally:
                    reset_current_user(token)
            return _orig_before(self, state, runtime)

        async def patched_abefore(self, state, runtime):
            uid = _resolve_user_id_from_runtime(runtime)
            if uid:
                token = set_current_user(_FakeUser(uid))
                try:
                    return await _orig_abefore(self, state, runtime)
                finally:
                    reset_current_user(token)
            return await _orig_abefore(self, state, runtime)

        DynamicContextMiddleware.before_agent = patched_before
        DynamicContextMiddleware.abefore_agent = patched_abefore
        _patch_logger.info(
            "Patched DynamicContextMiddleware: user_id now resolved from runtime.context"
        )

    _apply_middleware_patch()
except Exception as e:
    _pl = logging.getLogger(__name__ + ".middleware_patch")
    _pl.warning("Could not patch DynamicContextMiddleware: %s", e)
