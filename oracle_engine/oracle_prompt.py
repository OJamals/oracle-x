from openai import OpenAI
import os
from typing import Dict
import env_config
from oracle_engine.model_attempt_logger import log_attempt
import time

API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = env_config.get_openai_api_base() or os.environ.get("OPENAI_API_BASE", "https://api.githubcopilot.com/v1")
_PREFERRED_MODEL = env_config.get_openai_model()

client = OpenAI(api_key=API_KEY, base_url=API_BASE)
try:
    MODEL_NAME = env_config.resolve_model(client, _PREFERRED_MODEL, test=True)
except Exception as e:
    print(f"[WARN] Model resolution failed, using preferred '{_PREFERRED_MODEL}': {e}")
    MODEL_NAME = _PREFERRED_MODEL

def _iter_fallback_models(primary: str):
    try:
        fallback = env_config.get_fallback_models()
    except Exception:
        fallback = []
    ordered = [primary] + [m for m in fallback if m != primary]
    seen = set()
    for m in ordered:
        if m not in seen:
            seen.add(m)
            yield m

def generate_oracle_prompt(data: Dict) -> str:
    """
    Generate a detailed prompt for the Oracle-X LLM agent.
    Args:
        data (dict): Dictionary with keys: internals, options, sentiment, anomalies, risk_threshold.
    Returns:
        str: Prompt string for LLM.
    """
    return f"""
    You are ORACLE-X, an expert trading scenario engine. Analyze the following data and generate actionable, diverse, and high-confidence trade ideas:
    Market Internals: {data['internals']}
    Options Flow: {data['options']}
    Sentiment: {data['sentiment']}
    Anomalies: {data['anomalies']}

    1. Identify the 1–3 best trades for tomorrow, covering different tickers or strategies if possible.
    2. For each trade, provide:
        - Entry price (or range)
        - Optimal timing
        - Stop-loss
        - Profit target
        - Rationale (1–2 sentences)
        - Counter-signal
        - Scenario tree (base/bull/bear case with probabilities)
    3. Write a 5-sentence 'Tomorrow’s Tape' summarizing the market outlook.
    4. Confidence must be >= {data['risk_threshold'] * 100}% for each trade.
    5. Output must be valid JSON, with a 'trades' list and a 'tomorrows_tape' string. No markdown, no extra text.
    6. Encourage variety in trade types and tickers if possible.
    """

def get_oracle_playbook(prompt: str) -> str:
    for model in _iter_fallback_models(MODEL_NAME):
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are ORACLE-X."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_completion_tokens=1024
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                if model != MODEL_NAME:
                    print(f"[INFO] Fallback model '{model}' succeeded for oracle_prompt.")
                log_attempt("oracle_prompt", model, start_time=start, success=True, empty=False, error=None)
                return content
            log_attempt("oracle_prompt", model, start_time=start, success=False, empty=True, error=None)
            print(f"[DEBUG] Model {model} returned empty content – falling back (oracle_prompt).")
        except Exception as e:
            log_attempt("oracle_prompt", model, start_time=start, success=False, empty=False, error=str(e))
            print(f"[DEBUG] Model {model} failed (oracle_prompt): {e}")
            continue
    print("[WARN] All models failed for oracle_prompt.")
    return ""