from oracle_engine.dispatchers.llm_dispatcher import call_llm


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
            content = call_llm(
                model=model,
                messages=[
                    {"role": "system", "content": "You are ORACLE-X."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
                use_cache=False,
                retries=3
            )
            if content:
                if model != MODEL_NAME:
                    print(
                        f"[INFO] Fallback model '{model}' succeeded for oracle_prompt."
                    )
                log_attempt(
                    "oracle_prompt",
                    model,
                    start_time=start,
                    success=True,
                    empty=False,
                    error=None,
                )
                return content
            log_attempt(
                "oracle_prompt",
                model,
                start_time=start,
                success=False,
                empty=True,
                error=None,
            )
            print(
                f"[DEBUG] Model {model} returned empty content – falling back (oracle_prompt)."
            )
        except Exception as e:
            log_attempt(
                "oracle_prompt",
                model,
                start_time=start,
                success=False,
                empty=False,
                error=str(e),
            )
            print(f"[DEBUG] Model {model} failed (oracle_prompt): {e}")
            continue
    print("[WARN] All models failed for oracle_prompt.")
    return ""
