import os
import base64
from openai import OpenAI
from typing import Optional
from core.config import config
from oracle_engine.model_attempt_logger import log_attempt
import time

API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = config.model.openai_api_base or os.environ.get("OPENAI_API_BASE", "https://api.githubcopilot.com/v1")
MODEL_NAME = config.model.openai_model

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set. Please set it to run the application.")

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

def _iter_fallback_models(primary: str):
    try:
        fallback = config.model.fallback_models
    except Exception:
        fallback = []
    ordered = [primary] + [m for m in fallback if m != primary]
    seen = set()
    for m in ordered:
        if m not in seen:
            seen.add(m)
            yield m


# 🔑 TOOL: Real-time sentiment analyzer
def get_sentiment(text: str, model_name: str = MODEL_NAME) -> str:
    """Return sentiment label for given text using LLM with fallback."""
    for model in _iter_fallback_models(model_name):
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis engine."},
                    {"role": "user", "content": f"Analyze sentiment: \"{text}\""}
                ],
                max_completion_tokens=20
            )
            msg = (resp.choices[0].message.content or "").strip()
            if msg:
                if model != model_name:
                    print(f"[INFO] Fallback model '{model}' succeeded for sentiment.")
                log_attempt("sentiment", model, start_time=start, success=True, empty=False, error=None)
                return msg
            else:
                log_attempt("sentiment", model, start_time=start, success=False, empty=True, error=None)
                print(f"[DEBUG] Sentiment model {model} returned empty content – falling back.")
        except Exception as e:
            log_attempt("sentiment", model, start_time=start, success=False, empty=False, error=str(e))
            print(f"[DEBUG] Sentiment model {model} failed: {e}")
            continue
    print("[WARN] All models failed for sentiment analysis.")
    return "Unknown"

# 🔑 TOOL: Visual chart analyzer
import io
from PIL import Image


def analyze_chart(image_bytes: Optional[bytes], model_name: str = MODEL_NAME) -> str:
    """Analyze a chart image and summarize signals using LLM with fallback."""
    try:
        if isinstance(image_bytes, bytes):
            b64_str = base64.b64encode(image_bytes).decode("utf-8")
        elif isinstance(image_bytes, str):
            b64_str = image_bytes
        elif image_bytes is None:
            return "No chart provided."
        else:
            raise ValueError("image_bytes must be bytes, base64 string, or None")
    except Exception as e:
        print(f"[DEBUG] Chart input preparation failed: {e}")
        return "Unknown"

    for model in _iter_fallback_models(model_name):
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a chart analysis engine."},
                    {"role": "user", "content": f"Analyze this chart (base64): {b64_str}"}
                ],
                max_completion_tokens=60
            )
            msg = (resp.choices[0].message.content or "").strip()
            if msg:
                if model != model_name:
                    print(f"[INFO] Fallback model '{model}' succeeded for chart analysis.")
                log_attempt("chart_analysis", model, start_time=start, success=True, empty=False, error=None)
                return msg
            else:
                log_attempt("chart_analysis", model, start_time=start, success=False, empty=True, error=None)
                print(f"[DEBUG] Chart model {model} returned empty content – falling back.")
        except Exception as e:
            log_attempt("chart_analysis", model, start_time=start, success=False, empty=False, error=str(e))
            print(f"[DEBUG] Chart model {model} failed: {e}")
            continue
    print("[WARN] All models failed for chart analysis.")
    return "Unknown"
