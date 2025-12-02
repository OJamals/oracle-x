"""Oracle prompt helpers routed through the centralized LLM dispatcher."""

from typing import Dict

from core.config import config
from oracle_engine.dispatchers.llm_dispatcher import dispatch_chat

MODEL_NAME = config.model.openai_model


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
    """Generate an Oracle playbook via the unified dispatcher with built-in fallbacks."""
    result = dispatch_chat(
        messages=[
            {"role": "system", "content": "You are ORACLE-X."},
            {"role": "user", "content": prompt},
        ],
        model=MODEL_NAME,
        max_tokens=1024,
        temperature=0.3,
        task_type="analytical",
        purpose="oracle_prompt",
        retries=2,
        use_cache=False,
    )
    return result.content


__all__ = ["generate_oracle_prompt", "get_oracle_playbook"]
