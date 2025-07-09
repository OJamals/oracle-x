import os
import base64
from openai import OpenAI
from typing import Optional

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.githubcopilot.com/v1")

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
else:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set. Please set it to run the application.")

MODEL_NAME = "gpt-4.1-2025-04-14"


# 🔑 TOOL: Real-time sentiment analyzer
def get_sentiment(text: str, model_name: str = MODEL_NAME) -> str:
    """
    Return sentiment label for given text using LLM.
    Args:
        text (str): Text to analyze.
        model_name (str): LLM model name.
    Returns:
        str: Sentiment label.
    """
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a sentiment analysis engine."},
                {"role": "user", "content": f"Analyze sentiment: \"{text}\""}
            ],
            max_completion_tokens=20
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Sentiment analysis failed: {e}")
        return "Unknown"

# 🔑 TOOL: Visual chart analyzer
import io
from PIL import Image


def analyze_chart(image_bytes: Optional[bytes], model_name: str = MODEL_NAME) -> str:
    """
    Analyze a chart image and summarize signals using LLM.
    Args:
        image_bytes (Optional[bytes]): Base64-encoded image bytes or string.
        model_name (str): LLM model name.
    Returns:
        str: Chart analysis summary.
    """
    try:
        # Accept either bytes or base64 string
        if isinstance(image_bytes, bytes):
            b64_str = base64.b64encode(image_bytes).decode('utf-8')
        elif isinstance(image_bytes, str):
            b64_str = image_bytes
        elif image_bytes is None:
            return "No chart provided."
        else:
            raise ValueError("image_bytes must be bytes, base64 string, or None")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a chart analysis engine."},
                {"role": "user", "content": f"Analyze this chart (base64): {b64_str}"}
            ],
            max_completion_tokens=60
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Chart analysis failed: {e}")
        return "Unknown"
