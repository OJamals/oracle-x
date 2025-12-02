def clean_signals_for_llm(signals: dict, max_items: int = 5) -> dict:
    """
    Cleans and summarizes the signals dict to prevent prompt bloat and ensure only high-signal, deduplicated, and size-constrained data is sent to the LLM.
    - Truncates long lists/arrays to top N items
    - Deduplicates and filters empty/low-signal entries
    - Summarizes verbose text fields
    - Removes excessive whitespace and boilerplate
    """
    import re
    from collections.abc import Sequence

    def clean_text(text, max_len=400):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\x20-\x7E]', '', text)  # Remove non-ASCII
        # After removing non-ASCII, collapse whitespace again to remove any double spaces left behind
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > max_len:
            return text[:max_len] + '...'
        return text

    def clean_list(lst, key=None, max_items=max_items):
        if not isinstance(lst, Sequence) or isinstance(lst, str):
            return []
        seen = set()
        cleaned = []
        for item in lst:
            val = item if key is None else item.get(key, "")
            val = clean_text(str(val))
            if val and val not in seen:
                seen.add(val)
                cleaned.append(item)
            if len(cleaned) >= max_items:
                break
        return cleaned

    cleaned = {}
    # Clean each field specifically
    cleaned["market_internals"] = clean_text(str(signals.get("market_internals", "")), 600)
    cleaned["options_flow"] = clean_list(signals.get("options_flow", []), max_items=max_items)
    cleaned["dark_pools"] = clean_list(signals.get("dark_pools", []), max_items=max_items)
    cleaned["sentiment_web"] = clean_list(signals.get("sentiment_web", []), max_items=max_items)
    cleaned["sentiment_llm"] = clean_text(str(signals.get("sentiment_llm", "")), 600)
    cleaned["chart_analysis"] = clean_text(str(signals.get("chart_analysis", "")), 600)
    cleaned["earnings_calendar"] = clean_list(signals.get("earnings_calendar", []), max_items=max_items)
    # google_trends removed (disabled) – field omitted
    cleaned["yahoo_headlines"] = clean_list(signals.get("yahoo_headlines", []), key="headline" if signals.get("yahoo_headlines") and isinstance(signals["yahoo_headlines"], list) and signals["yahoo_headlines"] and isinstance(signals["yahoo_headlines"][0], dict) and "headline" in signals["yahoo_headlines"][0] else None, max_items=max_items)
    cleaned["finviz_breadth"] = clean_text(str(signals.get("finviz_breadth", "")), 400)
    cleaned["tickers"] = clean_list(signals.get("tickers", []), max_items=10)
    return cleaned
import re
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List


def extract_scenario_tree(llm_output: str, strict: bool = False) -> Optional[Dict[str, Any]]:
    """
    Extracts the scenario_tree dictionary from LLM output, handling various formats and malformed JSON.
    If strict=True, only accept valid dicts; else, fallback to best-effort extraction.
    Returns a dict if found, else None.
    """
    def try_json_parse(text):
        try:
            return json.loads(text)
        except Exception as e:
            print(f"[DEBUG] JSON parse failed: {e}")
            return None

    def extract_from_data(data):
        if not isinstance(data, dict):
            return None
        if 'scenario_tree' in data and isinstance(data['scenario_tree'], dict):
            print("[DEBUG] scenario_tree found via direct JSON parse.")
            return data['scenario_tree']
        if 'trades' in data and isinstance(data['trades'], list):
            for trade in data['trades']:
                if isinstance(trade, dict) and 'scenario_tree' in trade and isinstance(trade['scenario_tree'], dict):
                    print("[DEBUG] scenario_tree found in trade via JSON parse.")
                    return trade['scenario_tree']
        return None

    # 1. Try direct JSON parse
    data = try_json_parse(llm_output)
    result = extract_from_data(data) if data else None
    if result:
        return result

    # 2. Try to extract JSON block with scenario_tree
    for match in re.finditer(r'```json(.*?)```', llm_output, re.DOTALL | re.IGNORECASE):
        block = match.group(1)
        data = try_json_parse(block)
        result = extract_from_data(data) if data else None
        if result:
            print("[DEBUG] scenario_tree found in markdown JSON block.")
            return result

    # 3. Try regex for scenario_tree dict pattern
    dict_pattern = r'"scenario_tree"\s*:\s*({.*?})'
    match = re.search(dict_pattern, llm_output, re.DOTALL)
    if match:
        try:
            return _extracted_from_extract_scenario_tree_42(match, "[DEBUG] scenario_tree found via regex dict pattern.")
        except Exception as e:
            print(f"[DEBUG] Regex dict pattern parse failed: {e}")

    # 4. Try to extract from any code block containing scenario_tree
    code_block = re.search(r'```[a-zA-Z]*\n(.*?scenario_tree.*?\n.*?)```', llm_output, re.DOTALL)
    if code_block:
        block = code_block[1]
        match = re.search(dict_pattern, block, re.DOTALL)
        if match:
            try:
                return _extracted_from_extract_scenario_tree_42(match, "[DEBUG] scenario_tree found in markdown code block.")
            except Exception as e:
                print(f"[DEBUG] Markdown code block parse failed: {e}")

    # 5. Fallback: try to extract a dict-like string and parse it
    if not strict:
        dict_like = re.search(r'\{[\s\S]*?\}', llm_output)
        if dict_like:
            try:
                candidate = dict_like[0].replace("'", '"')
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and all(k in parsed for k in ("base_case", "bull_case", "bear_case")):
                    print("[DEBUG] scenario_tree fallback dict-like parse succeeded.")
                    return parsed
            except Exception as e:
                print(f"[DEBUG] Fallback dict-like parse failed: {e}")

    print("[DEBUG] scenario_tree not found in LLM output.")
    return None


# TODO Rename this here and in `extract_scenario_tree`
def _extracted_from_extract_scenario_tree_42(match, arg1) -> Dict[str, Any]:
    """
    Helper to parse scenario_tree from regex match.
    Args:
        match: Regex match object containing the scenario_tree string.
        arg1: Debug message to print.
    Returns:
        dict: Parsed scenario_tree dictionary.
    """
    scenario_tree_str = match.group(1)
    scenario_tree_str = scenario_tree_str.replace("'", '"')
    scenario_tree = json.loads(scenario_tree_str)
    print(arg1)
    return scenario_tree
from openai import OpenAI
from vector_db.local_store import query_similar
from vector_db.prompt_booster import build_boosted_prompt, batch_build_boosted_prompts

# 🧩 Import your local scraper modules
from data_feeds.market_internals import fetch_market_internals
from data_feeds.options_flow import fetch_options_flow
from data_feeds.dark_pools import fetch_dark_pool_data
from data_feeds.sentiment import fetch_sentiment_data
from data_feeds.earnings_calendar import fetch_earnings_calendar
from oracle_engine.tools import get_sentiment, analyze_chart
# (Legacy) twitter sentiment import retained if needed elsewhere
from data_feeds.twitter_sentiment import fetch_twitter_sentiment
from data_feeds.news_scraper import fetch_headlines_yahoo_finance
from data_feeds.finviz_scraper import fetch_finviz_breadth
from data_feeds.ticker_universe import fetch_ticker_universe

# Ticker validation is optional; fall back to a no-op if the optimizer module
# is unavailable in the runtime environment.
try:
    from optimizations.ticker_validator import validate_tickers
except Exception:  # pragma: no cover - optional dependency
    def validate_tickers(tickers):  # type: ignore
        return tickers

from core.config import config

API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = config.model.openai_api_base or os.environ.get("OPENAI_API_BASE", "https://api.githubcopilot.com/v1")
MODEL_NAME = config.model.openai_model

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

def get_signals_from_scrapers(prompt_text: str, chart_image_b64: str, enable_premium: bool = True) -> Dict[str, Any]:
    """
    Pulls real-time signals from FREE data sources + optional PREMIUM unique data.
    
    FREE Data Sources (Core):
    - Market Internals: yfinance (indices, VIX, breadth)
    - Options Flow: yfinance (unusual volume analysis)
    - Dark Pools: yfinance (volume anomaly detection) + SYNTHETIC signals from options correlation
    - Sentiment: Reddit + Twitter (direct APIs, no orchestrator)
    - Earnings: yfinance (income statements)
    - News: Yahoo Finance (web scraping)
    - Breadth: Finviz (web scraping)
    - LLM Sentiment: Local analysis
    - Chart Analysis: Local vision model
    
    PREMIUM Data Sources (Optional - UNIQUE data only):
    - Finnhub: Insider sentiment, analyst recommendations, price targets
    - FMP: Analyst estimates, institutional ownership, DCF valuations
    
    Note: Premium APIs only fetch data NOT available from free sources
    
    Args:
        prompt_text (str): User prompt or market summary.
        chart_image_b64 (str): Base64-encoded chart image.
        enable_premium (bool): Enable premium API calls for unique data (default: True)
    Returns:
        dict: All signals snapshot from free + optional premium sources.
    """
    # Fetch ticker universe (top 40 tickers)
    tickers = fetch_ticker_universe(sample_size=40)
    
    # Core market data (yfinance - FREE)
    internals = fetch_market_internals()
    options_flow = fetch_options_flow(tickers)
    
    # Dark pool detection with synthetic signal enhancement
    real_dark_pools_dict = fetch_dark_pool_data(tickers)
    real_dark_pools_list = real_dark_pools_dict.get('dark_pools', []) if isinstance(real_dark_pools_dict, dict) else []
    
    # Generate synthetic dark pool signals from options flow correlation
    try:
        from data_feeds.synthetic_darkpool_signals import generate_synthetic_darkpool_signals
        dark_pools_list = generate_synthetic_darkpool_signals(
            options_data=options_flow,
            real_darkpool_data=real_dark_pools_list
        )
        if dark_pools_list and len(dark_pools_list) > 0:
            print(f"[INFO] Enhanced dark pool detection: {len(dark_pools_list)} signals generated")
            # Return dict in same format as fetch_dark_pool_data
            dark_pools = {
                "dark_pools": dark_pools_list,
                "data_source": "synthetic_options_correlation",
                "timestamp": datetime.now().isoformat(),
                "total_detected": len(dark_pools_list)
            }
        else:
            dark_pools = real_dark_pools_dict
            print(f"[INFO] Using baseline dark pool detection: {len(real_dark_pools_list)} signals")
    except Exception as e:
        import traceback
        print(f"[WARN] Synthetic dark pool signal generation failed: {e}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        dark_pools = real_dark_pools_dict
    earnings = fetch_earnings_calendar(tickers)
    
    # Sentiment data (Reddit + Twitter - FREE)
    # fetch_sentiment_data internally calls reddit_sentiment and twitter_sentiment
    sentiment_web = fetch_sentiment_data(tickers)
    
    # News and breadth (web scraping - FREE)
    yahoo_headlines = fetch_headlines_yahoo_finance()
    finviz_breadth = fetch_finviz_breadth()
    
    # LLM-based analysis (local - FREE)
    sentiment_llm = get_sentiment(prompt_text, model_name=MODEL_NAME)
    
    # Chart analysis (local vision model - FREE)
    import base64
    chart_bytes = None
    if chart_image_b64:
        try:
            chart_bytes = base64.b64decode(chart_image_b64)
        except Exception as e:
            print(f"[DEBUG] Failed to decode chart_image_b64: {e}")
    chart_analysis = analyze_chart(chart_bytes, model_name=MODEL_NAME)
    
    # Optional: Premium unique data (if API keys available)
    premium_data = None
    if enable_premium:
        try:
            from data_feeds.strategic_premium_feeds import get_premium_feeds
            premium = get_premium_feeds()
            
            # Only fetch if at least one API is available
            if premium.finnhub_available or premium.fmp_available:
                # Limit to top 5 tickers to avoid rate limits
                top_tickers = tickers[:5]
                print(f"[INFO] Fetching premium unique data for {len(top_tickers)} tickers...")
                premium_data = premium.get_all_premium_data(top_tickers, max_symbols=5)
                print(f"[INFO] Premium data fetched successfully")
        except Exception as e:
            print(f"[WARN] Premium data fetch failed (non-critical): {e}")
    
    result = {
        "tickers": tickers,
        "market_internals": internals,
        "options_flow": options_flow,
        "dark_pools": dark_pools,
        "sentiment_web": sentiment_web,  # Contains reddit + twitter data
        "sentiment_llm": sentiment_llm,
        "chart_analysis": chart_analysis,
        "earnings_calendar": earnings,
        "yahoo_headlines": yahoo_headlines,
        "finviz_breadth": finviz_breadth,
    }
    
    # Add premium data if available
    if premium_data:
        result["premium_data"] = premium_data
    
    return result

def pull_similar_scenarios(thesis: str) -> str:
    """
    Query local vector store for similar past scenarios to enrich the scenario tree.
    Returns a formatted string of similar scenarios.
    """
    hits = query_similar(thesis, top_k=3)
    if not hits:
        return "None found."
    text = ""
    for hit in hits:
        payload = hit.get('payload', {})
        if payload:
            text += (
                f"- Ticker: {payload.get('ticker', 'N/A')}, "
                f"Direction: {payload.get('direction', 'N/A')}, "
                f"Thesis: {payload.get('thesis', 'N/A')}, "
                f"Date: {payload.get('date', 'N/A')}\n"
            )
    return text.strip()

def _iter_fallback_models(primary: str) -> List[str]:
    """Return ordered list of models to try: primary then configured fallbacks (deduplicated)."""
    try:
        fallback = config.model.fallback_models
    except Exception:
        fallback = []
    ordered = [primary] + [m for m in fallback if m != primary]
    # de‑dupe while preserving order
    seen = set()
    result: List[str] = []
    for m in ordered:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result

from oracle_engine.model_attempt_logger import log_attempt
import time

def adjust_scenario_tree(signals: Dict[str, Any], similar_scenarios: str, model_name: str = MODEL_NAME) -> str:
    """
    Calls the LLM to produce an adjusted scenario tree with improved probabilities.
    Returns the LLM's response as a string.
    """
    prompt = f"""
Given these live signals:
Market Internals: {signals['market_internals']}
Options Flow: {signals['options_flow']}
Dark Pools: {signals['dark_pools']}
Sentiment (Web): {signals['sentiment_web']}
Sentiment (LLM): {signals['sentiment_llm']}
Chart Analysis: {signals['chart_analysis']}
Earnings Calendar: {signals['earnings_calendar']}

And these similar past scenarios:
{similar_scenarios}

Analyze and output a scenario tree with updated probabilities for base/bull/bear cases.
Explain how the past scenarios influence your adjustments.
""".strip()

    for model in _iter_fallback_models(model_name):
        print(f"[DEBUG] Trying model: {model}")
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are ORACLE-X, an adaptive scenario engine."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=600
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                if model != model_name:
                    print(f"[INFO] Fallback model '{model}' succeeded for adjust_scenario_tree.")
                else:
                    print(f"[DEBUG] Model {model} returned non-empty response.")
                log_attempt("adjust_scenario_tree", model, start_time=start, success=True, empty=False, error=None)
                return content
            else:
                log_attempt("adjust_scenario_tree", model, start_time=start, success=False, empty=True, error=None)
                print(f"[DEBUG] Model {model} returned empty content – falling back.")
        except Exception as e:
            log_attempt("adjust_scenario_tree", model, start_time=start, success=False, empty=False, error=str(e))
            print(f"[DEBUG] Model {model} failed: {e}")
            continue
    print("[WARN] All models returned empty or failed for adjust_scenario_tree.")
    return ""

def adjust_scenario_tree_with_boost(signals: Dict[str, Any], similar_scenarios: str, model_name: str = MODEL_NAME) -> str:
    """
    Calls the LLM to produce an adjusted scenario tree, boosting the prompt with similar scenarios from ChromaDB.
    Returns the LLM's response as a string.
    """
    base_prompt = f"""
Given these live signals:
Market Internals: {signals['market_internals']}
Options Flow: {signals['options_flow']}
Dark Pools: {signals['dark_pools']}
Sentiment (Web): {signals['sentiment_web']}
Sentiment (LLM): {signals['sentiment_llm']}
Chart Analysis: {signals['chart_analysis']}
Earnings Calendar: {signals['earnings_calendar']}

CRITICAL: Dark Pool Signal Interpretation Guidelines
When analyzing dark pool signals, consider the data source and confidence level:

1. SOURCE: 'volume_analysis_proxy' (traditional method)
   - Confidence: 30-40% accuracy
   - Use as supporting evidence only

2. SOURCE: 'synthetic_options_correlation' (enhanced method)
   - Confidence: 50-65% accuracy  
   - Generated from large options sweeps (volume >50k, V/OI ratio >3.0)
   - When dark_pool_probability > 0.7 AND confidence > 0.65: HIGH INSTITUTIONAL ACTIVITY
   - When dark_pool_probability > 0.5 AND confidence > 0.5: ELEVATED INSTITUTIONAL ACTIVITY
   - institutional_action = 'BUY' → Increase bullish scenario probability by 5-10%
   - institutional_action = 'SELL' → Increase bearish scenario probability by 5-10%

3. SOURCE: 'synthetic_validated_by_real' (highest confidence)
   - Confidence: 70-80% accuracy
   - Synthetic signal confirmed by real volume data
   - Weight most heavily in scenario probability adjustments

4. SOURCE: 'finra_ats' or 'polygon_realtime' (when available)
   - Confidence: 80-95% accuracy
   - Official institutional flow data
   - Use as primary evidence for scenario weighting

IMPORTANT: Always mention dark pool signals in your reasoning when available and explain how they influenced your probability adjustments.

And these similar past scenarios:
{similar_scenarios}

Analyze and output a scenario tree with updated probabilities for base/bull/bear cases.
Explain how the past scenarios influence your adjustments.
""".strip()
        
    # Boost the prompt with ChromaDB recall
    boosted_prompt = build_boosted_prompt(base_prompt, str(signals.get('chart_analysis', '')))
    for model in _iter_fallback_models(model_name):
        print(f"[DEBUG] Trying model: {model} (boosted)")
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are ORACLE-X, an adaptive scenario engine."},
                    {"role": "user", "content": boosted_prompt}
                ],
                max_completion_tokens=600
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                if model != model_name:
                    print(f"[INFO] Fallback model '{model}' succeeded for adjust_scenario_tree_with_boost.")
                else:
                    print(f"[DEBUG] Model {model} returned non-empty response (boosted).")
                log_attempt("adjust_scenario_tree_with_boost", model, start_time=start, success=True, empty=False, error=None)
                return content
            else:
                log_attempt("adjust_scenario_tree_with_boost", model, start_time=start, success=False, empty=True, error=None)
                print(f"[DEBUG] Model {model} returned empty content (boosted) – falling back.")
        except Exception as e:
            log_attempt("adjust_scenario_tree_with_boost", model, start_time=start, success=False, empty=False, error=str(e))
            print(f"[DEBUG] Model {model} failed (boosted): {e}")
            continue
    print("[WARN] All models returned empty or failed (boosted).")
    return ""

def batch_adjust_scenario_trees_with_boost(
    signals_list: List[Dict[str, Any]],
    similar_scenarios_list: List[str],
    model_name: str = MODEL_NAME
) -> List[str]:
    """
    Batch version: Calls the LLM to produce adjusted scenario trees for multiple prompts, boosting each with ChromaDB recall.
    Returns a list of LLM responses (one per prompt).
    """
    base_prompts = []
    for signals, similar_scenarios in zip(signals_list, similar_scenarios_list):
        base_prompt = f"""
Given these live signals:
Market Internals: {signals['market_internals']}
Options Flow: {signals['options_flow']}
Dark Pools: {signals['dark_pools']}
Sentiment (Web): {signals['sentiment_web']}
Sentiment (LLM): {signals['sentiment_llm']}
Chart Analysis: {signals['chart_analysis']}
Earnings Calendar: {signals['earnings_calendar']}

And these similar past scenarios:
{similar_scenarios}

Analyze and output a scenario tree with updated probabilities for base/bull/bear cases.
Explain how the past scenarios influence your adjustments.
""".strip()
        base_prompts.append(base_prompt)
    # Use batch prompt boosting
    trade_theses = [str(s.get('chart_analysis', '')) for s in signals_list]
    boosted_prompts = batch_build_boosted_prompts(base_prompts, trade_theses)
    results = []
    for boosted_prompt in boosted_prompts:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are ORACLE-X, an adaptive scenario engine."},
                    {"role": "user", "content": boosted_prompt}
                ],
                max_completion_tokens=600
            )
            content = resp.choices[0].message.content
            if content is not None:
                content = content.strip()
                if content:
                    results.append(content)
                else:
                    results.append("")
            else:
                results.append("")
        except Exception as e:
            print(f"[DEBUG] Batch model {model_name} failed: {e}")
            results.append("")
    return results

def generate_final_playbook(signals, scenario_tree, model_name=MODEL_NAME):
    """
    Final LLM step: generates tomorrow's best trades + 'Tomorrow's Tape'.
    """


    # Clean signals before prompt construction
    signals = clean_signals_for_llm(signals)

    # Get real-time prices for ALL tickers using orchestrator data
    current_prices = {}
    all_tickers = set()
    
    # Extract all tickers from signals
    if 'tickers' in signals and signals['tickers']:
        all_tickers.update(signals['tickers'])
    
    # Extract tickers from scenario tree
    if scenario_tree:
        import re
        scenario_tickers = re.findall(r'\b[A-Z]{2,5}\b', scenario_tree)
        all_tickers.update([t for t in scenario_tickers if len(t) <= 5 and t.isalpha()])

    # Validate the merged ticker list to strip out verbs or other uppercase words
    # (e.g., SELL) that may appear in the LLM JSON but are not real symbols.
    validated_tickers = []
    if all_tickers:
        try:
            validated_tickers = validate_tickers(sorted(all_tickers))
        except Exception:
            validated_tickers = sorted(all_tickers)
    else:
        validated_tickers = []
    
    # Get orchestrator for real-time pricing
    try:
        from data_feeds.data_feed_orchestrator import get_orchestrator
        orch = get_orchestrator()
        
        for ticker in validated_tickers:
            try:
                # Get current quote from orchestrator
                quote = orch.get_quote(ticker)
                if quote and hasattr(quote, 'price') and quote.price:
                    current_prices[ticker] = f"{float(quote.price):.2f}"
                else:
                    # Fallback to market data
                    market_data = orch.get_market_data(ticker, period="1d", interval="1d")
                    if market_data and hasattr(market_data, 'data') and not market_data.data.empty:
                        latest_close = market_data.data['Close'].iloc[-1]
                        current_prices[ticker] = f"{float(latest_close):.2f}"
            except Exception as e:
                # Skip tickers that fail to fetch
                continue
                
    except Exception as e:
        print(f"[WARN] Could not access orchestrator for pricing: {e}")
        pass

    price_context = ""
    if current_prices:
        price_context = f"""
CRITICAL - CURRENT MARKET PRICES (Use these exact prices, NOT training data):
{chr(10).join([f"{ticker}: ${price}" for ticker, price in current_prices.items()])}

MANDATORY: All entry ranges, profit targets, and stop losses MUST be based on these current prices, not historical training data.
"""

    prompt = f"""
Signals:
Market Internals: {signals['market_internals']}
Options Flow: {signals['options_flow']}
Dark Pools: {signals['dark_pools']}
Sentiment (Web): {signals['sentiment_web']}
Sentiment (LLM): {signals['sentiment_llm']}
Chart Analysis: {signals['chart_analysis']}
Earnings Calendar: {signals['earnings_calendar']}
{price_context}
Adjusted Scenario Tree:
{scenario_tree}

Based on this, generate the 1–3 highest-confidence trades for tomorrow.
Include: ticker, direction, instrument, entry range, profit target, stop-loss,
counter-signal, and a 5-sentence 'Tomorrow's Tape'.

Format your response as **valid JSON only** (no markdown, no extra text, no comments) with:
- 'trades': a list of trade objects
- 'tomorrows_tape': a string summary

IMPORTANT: For each trade, include a 'thesis' field summarizing the core rationale in 1-2 sentences.
IMPORTANT: For each trade, include a 'scenario_tree' field with a dictionary of scenario probabilities (base_case, bull_case, bear_case), e.g.:
"scenario_tree": {{"base_case": "70% - ...", "bull_case": "20% - ...", "bear_case": "10% - ..."}}
If you are unsure, make a reasonable estimate. Do NOT omit the 'scenario_tree' field.

---
EXAMPLE RESPONSE (strictly follow this structure):
{{
  "trades": [
    {{
      "ticker": "AAPL",
      "direction": "long",
      "instrument": "shares",
      "entry_range": "190-192",
      "profit_target": "198",
      "stop_loss": "187",
      "counter_signal": "Break below 187",
      "thesis": "Apple is showing strong momentum after earnings.",
      "scenario_tree": {{
        "base_case": "70% - Continues up on strong demand.",
        "bull_case": "20% - Explosive move if market rallies.",
        "bear_case": "10% - Drops if market reverses."
      }}
    }}
  ],
  "tomorrows_tape": "Markets are poised for a breakout as tech leads..."
}}
---

DO NOT include any text before or after the JSON. Output only the JSON object.
""".strip()

    for model in _iter_fallback_models(model_name):
        print(f"[DEBUG] Trying model: {model}")
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are ORACLE-X, the final Playbook composer."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1024
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                if model != model_name:
                    print(f"[INFO] Fallback model '{model}' succeeded for generate_final_playbook.")
                else:
                    print(f"[DEBUG] Model {model} returned non-empty response.")
                log_attempt("generate_final_playbook", model, start_time=start, success=True, empty=False, error=None)
                return content
            else:
                log_attempt("generate_final_playbook", model, start_time=start, success=False, empty=True, error=None)
                print(f"[DEBUG] Model {model} returned empty content – falling back.")
        except Exception as e:
            log_attempt("generate_final_playbook", model, start_time=start, success=False, empty=False, error=str(e))
            print(f"[DEBUG] Model {model} failed: {e}")
            continue
    print("[WARN] All models returned empty or failed for generate_final_playbook.")
    return ""
