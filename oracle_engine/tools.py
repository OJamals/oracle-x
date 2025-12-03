"""LLM-backed utility helpers routed through the centralized dispatcher."""

from __future__ import annotations

import base64
from typing import Optional

from core.config import config, get_openai_model
from oracle_engine.dispatchers.llm_dispatcher import dispatch_chat

MODEL_NAME = get_openai_model()
MODEL_NAME = config.model.get_openai_model()


def get_sentiment(text: str, model_name: str = MODEL_NAME) -> str:
    """Return sentiment label for given text using the LLM dispatcher."""
    result = dispatch_chat(
        messages=[
            {"role": "system", "content": "You are a sentiment analysis engine."},
            {"role": "user", "content": f'Analyze sentiment: "{text}"'},
        ],
        model=model_name,
        max_tokens=20,
        temperature=0.3,
        task_type="analytical",
        purpose="sentiment",
        retries=2,
    )
    return result.content or "Unknown"


def analyze_chart(image_bytes: Optional[bytes], model_name: str = MODEL_NAME) -> str:
    """Analyze a chart image and summarize signals using the dispatcher."""
    try:
        if isinstance(image_bytes, bytes):
            b64_str = base64.b64encode(image_bytes).decode("utf-8")
        elif isinstance(image_bytes, str):
            b64_str = image_bytes
        elif image_bytes is None:
            return "No chart provided."
        else:
            raise ValueError("image_bytes must be bytes, base64 string, or None")
    except Exception as exc:
        print(f"[DEBUG] Chart input preparation failed: {exc}")
        return "Unknown"

    result = dispatch_chat(
        messages=[
            {"role": "system", "content": "You are a chart analysis engine."},
            {"role": "user", "content": f"Analyze this chart (base64): {b64_str}"},
        ],
        model=model_name,
        max_tokens=60,
        temperature=0.3,
        task_type="analytical",
        purpose="chart_analysis",
        retries=2,
        use_cache=True,
    )
    return result.content or "Unknown"


__all__ = ["get_sentiment", "analyze_chart"]
