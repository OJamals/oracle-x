"""Lightweight in-process logging of model invocation attempts.

Each attempt record structure:
{
  'purpose': str,          # high-level step (e.g. 'adjust_scenario_tree')
  'model': str,            # model name attempted
  'success': bool,         # True if non-empty content returned
  'empty': bool,           # True if call succeeded but empty content
  'error': str | None,     # Error message (truncated) if exception raised
  'latency_sec': float     # Wall clock latency in seconds
}

The list is appended to by model calling code and later drained (pop_attempts)
by the pipeline to persist into the final playbook metadata.
"""

from __future__ import annotations
import time
from typing import List, Dict, Any

_MODEL_ATTEMPTS: List[Dict[str, Any]] = []


def log_attempt(
    purpose: str,
    model: str,
    start_time: float,
    *,
    success: bool,
    empty: bool,
    error: str | None,
):
    latency = time.time() - start_time
    _MODEL_ATTEMPTS.append(
        {
            "purpose": purpose,
            "model": model,
            "success": success,
            "empty": empty,
            "error": (error[:160] if error else None),
            "latency_sec": round(latency, 4),
        }
    )


def pop_attempts() -> List[Dict[str, Any]]:
    """Return and clear accumulated attempt records."""
    global _MODEL_ATTEMPTS
    attempts = list(_MODEL_ATTEMPTS)
    _MODEL_ATTEMPTS = []
    return attempts


def get_attempts_snapshot() -> List[Dict[str, Any]]:
    return list(_MODEL_ATTEMPTS)
