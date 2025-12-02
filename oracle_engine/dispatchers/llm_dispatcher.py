"""
Centralized LLM dispatcher with caching, retries, and config-driven model selection.

This module wraps the multi-provider LLM client so callers do not need to duplicate
fallback handling or caching logic. All chat completions should flow through here
to ensure consistent telemetry and cache behavior.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from core.config import config
from core.unified_cache_manager import cache_manager
from oracle_engine.llm_client import get_llm_client
from oracle_engine.model_attempt_logger import log_attempt

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL_SECONDS = 300
DEFAULT_RETRIES = 1


@dataclass
class LLMDispatchResult:
    """Normalized response returned by the dispatcher."""

    content: str
    model: str
    provider: Optional[str]
    cached: bool = False
    tokens_used: Optional[Dict[str, int]] = None
    error: Optional[str] = None


def _iter_models(primary: str) -> List[str]:
    """Yield primary model followed by configured fallbacks, de-duplicated."""
    fallback = getattr(getattr(config, "model", None), "fallback_models", []) or []
    ordered = [primary] + [m for m in fallback if m != primary]
    seen = set()
    models: List[str] = []
    for model in ordered:
        if model not in seen:
            seen.add(model)
            models.append(model)
    return models


def _normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Ensure messages are JSON-serializable and deterministic for hashing."""
    normalized: List[Dict[str, str]] = []
    for message in messages:
        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
        )
    return normalized


def _make_cache_key(
    messages: Sequence[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    task_type: str,
    purpose: str,
    cache_key: Optional[str],
) -> str:
    if cache_key:
        return cache_key

    payload = {
        "messages": _normalize_messages(messages),
        "model": model,
        "temperature": round(temperature, 4),
        "max_tokens": max_tokens,
        "task_type": task_type,
        "purpose": purpose,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"llm:{digest}"


class LLMDispatcher:
    """Dispatch chat completion requests with caching and retries."""

    def __init__(self):
        self.client = get_llm_client()
        ttl_minutes = (
            getattr(getattr(config, "data_feeds", None), "cache_ttl_minutes", 5) or 5
        )
        self.default_ttl = max(int(ttl_minutes * 60), DEFAULT_CACHE_TTL_SECONDS)
        self.default_model = getattr(
            getattr(config, "model", None), "openai_model", "gpt-4o"
        )

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        task_type: str = "general",
        purpose: str = "llm_call",
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        use_cache: bool = True,
        retries: int = DEFAULT_RETRIES,
    ) -> LLMDispatchResult:
        """Execute a chat completion with caching and fallback models."""
        chosen_model = model or self.default_model
        resolved_cache_key = _make_cache_key(
            messages,
            chosen_model,
            temperature,
            max_tokens,
            task_type,
            purpose,
            cache_key,
        )

        if use_cache:
            cached_value = cache_manager.get(resolved_cache_key)
            if isinstance(cached_value, dict) and cached_value.get("content"):
                return LLMDispatchResult(
                    content=cached_value.get("content", ""),
                    model=cached_value.get("model", chosen_model),
                    provider=cached_value.get("provider"),
                    cached=True,
                    tokens_used=cached_value.get("tokens_used"),
                    error=None,
                )

        last_error: Optional[str] = None
        for candidate_model in _iter_models(chosen_model):
            result = self._call_model(
                candidate_model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                task_type=task_type,
                purpose=purpose,
                retries=retries,
            )
            if result.content:
                if use_cache:
                    cache_manager.set(
                        resolved_cache_key,
                        {
                            "content": result.content,
                            "model": result.model,
                            "provider": result.provider,
                            "tokens_used": result.tokens_used,
                        },
                        ttl=cache_ttl or self.default_ttl,
                    )
                return result
            last_error = result.error

        return LLMDispatchResult(
            content="",
            model=chosen_model,
            provider=None,
            cached=False,
            tokens_used=None,
            error=last_error,
        )

    def _call_model(
        self,
        model: str,
        messages: Sequence[Dict[str, Any]],
        *,
        temperature: float,
        max_tokens: Optional[int],
        task_type: str,
        purpose: str,
        retries: int,
    ) -> LLMDispatchResult:
        """Call a specific model with retry/backoff."""
        last_error: Optional[str] = None
        response_provider: Optional[str] = None
        tokens_used: Optional[Dict[str, int]] = None

        for attempt in range(retries + 1):
            start = time.time()
            try:
                response = self.client.create_chat_completion(
                    messages=list(messages),
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    task_type=task_type,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                last_error = str(exc)
                log_attempt(
                    purpose,
                    model,
                    start_time=start,
                    success=False,
                    empty=False,
                    error=last_error,
                )
                if attempt < retries:
                    time.sleep(min(0.5 * (attempt + 1), 2.0))
                continue

            response_provider = getattr(
                getattr(response, "provider_used", None), "value", None
            )
            tokens_used = getattr(response, "tokens_used", None)

            success = getattr(response, "success", False) and bool(
                getattr(response, "content", "").strip()
            )
            empty = getattr(response, "success", False) and not success
            log_attempt(
                purpose,
                model,
                start_time=start,
                success=success,
                empty=empty,
                error=getattr(response, "error_message", None),
            )

            if success:
                content = (response.content or "").strip()
                return LLMDispatchResult(
                    content=content,
                    model=getattr(response, "model_used", model) or model,
                    provider=response_provider,
                    cached=False,
                    tokens_used=tokens_used,
                    error=None,
                )

            last_error = getattr(response, "error_message", None) or "empty response"
            if attempt < retries:
                time.sleep(min(0.5 * (attempt + 1), 2.0))

        return LLMDispatchResult(
            content="",
            model=model,
            provider=response_provider,
            cached=False,
            tokens_used=tokens_used,
            error=last_error,
        )


_dispatcher = LLMDispatcher()


def dispatch_chat(*args: Any, **kwargs: Any) -> LLMDispatchResult:
    """Module-level convenience wrapper for synchronous dispatch."""
    return _dispatcher.chat(*args, **kwargs)


async def dispatch_chat_async(*args: Any, **kwargs: Any) -> LLMDispatchResult:
    """Async wrapper to use dispatcher from asyncio code paths."""
    loop = asyncio.get_running_loop()
    func = functools.partial(dispatch_chat, *args, **kwargs)
    return await loop.run_in_executor(None, func)


def call_llm(*args: Any, **kwargs: Any) -> str:
    """Legacy function that returns content string directly."""
    result = dispatch_chat(*args, **kwargs)
    return result.content


__all__ = [
    "LLMDispatchResult",
    "dispatch_chat",
    "dispatch_chat_async",
    "LLMDispatcher",
    "call_llm",
]
