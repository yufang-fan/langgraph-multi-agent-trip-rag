"""Small JSON cache with optional Redis and an always-available memory fallback."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..config import get_settings

logger = logging.getLogger(__name__)

_memory_cache: dict[str, tuple[float, Any]] = {}
_redis_client: Any | None = None
_redis_unavailable_logged = False


def _get_redis_client():
    """Return a Redis client only when explicitly enabled and importable."""
    global _redis_client
    global _redis_unavailable_logged

    settings = get_settings()
    if not settings.rag_redis_enabled:
        return None

    try:
        import redis
    except ImportError:
        if not _redis_unavailable_logged:
            logger.warning("Redis is enabled but the redis package is not installed.")
            _redis_unavailable_logged = True
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        client = redis.Redis.from_url(settings.rag_redis_url, decode_responses=True)
        client.ping()
        _redis_client = client
        return client
    except Exception as exc:
        if not _redis_unavailable_logged:
            logger.warning("Redis connection failed, using memory cache: %s", exc)
            _redis_unavailable_logged = True
        return None


def get_cached_json(key: str) -> Any | None:
    """Get a JSON value from Redis or the local memory fallback."""
    cached = _memory_cache.get(key)
    if cached is not None:
        expires_at, value = cached
        if expires_at > time.time():
            return value
        _memory_cache.pop(key, None)

    client = _get_redis_client()
    if client is None:
        return None

    try:
        raw_value = client.get(key)
        if raw_value is None:
            return None
        return json.loads(raw_value)
    except Exception as exc:
        logger.debug("Read cache failed for %s: %s", key, exc)
        return None


def set_cached_json(key: str, value: Any, expire_seconds: int = 21600) -> None:
    """Write a JSON value to Redis and the local memory fallback."""
    _memory_cache[key] = (time.time() + expire_seconds, value)

    client = _get_redis_client()
    if client is None:
        return

    try:
        client.set(key, json.dumps(value, ensure_ascii=False), ex=expire_seconds)
    except Exception as exc:
        logger.debug("Write cache failed for %s: %s", key, exc)
