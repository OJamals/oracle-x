from openai import OpenAI
import os
from typing import Dict

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.githubcopilot.com/v1")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

MODEL_NAME = "gpt-4.1-2025-04-14"

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
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are ORACLE-X."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_completion_tokens=1024
    )
    return resp.choices[0].message.content.strip()