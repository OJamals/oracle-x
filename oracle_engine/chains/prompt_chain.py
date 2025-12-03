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
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\x20-\x7E]", "", text)  # Remove non-ASCII
        # After removing non-ASCII, collapse whitespace again to remove any double spaces left behind
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_len:
            return text[:max_len] + "..."
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
    cleaned["market_internals"] = clean_text(
        str(signals.get("market_internals", "")), 600
    )
    cleaned["options_flow"] = clean_list(
        signals.get("options_flow", []), max_items=max_items
    )
    cleaned["dark_pools"] = clean_list(
        signals.get("dark_pools", []), max_items=max_items
    )
    cleaned["sentiment_web"] = clean_list(
        signals.get("sentiment_web", []), max_items=max_items
    )
    cleaned["sentiment_llm"] = clean_text(str(signals.get("sentiment_llm", "")), 600)
    cleaned["chart_analysis"] = clean_text(str(signals.get("chart_analysis", "")), 600)
    cleaned["earnings_calendar"] = clean_list(
        signals.get("earnings_calendar", []), max_items=max_items
    )
    # google_trends removed (disabled) – field omitted
    cleaned["yahoo_headlines"] = clean_list(
        signals.get("yahoo_headlines", []),
        key=(
            "headline"
            if signals.get("yahoo_headlines")
            and isinstance(signals["yahoo_headlines"], list)
            and signals["yahoo_headlines"]
            and isinstance(signals["yahoo_headlines"][0], dict)
            and "headline" in signals["yahoo_headlines"][0]
            else None
        ),
        max_items=max_items,
    )
    cleaned["finviz_breadth"] = clean_text(str(signals.get("finviz_breadth", "")), 400)
    cleaned["tickers"] = clean_list(signals.get("tickers", []), max_items=10)
    return cleaned


import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.config import config
from oracle_engine.dispatchers.llm_dispatcher import dispatch_chat
from oracle_engine.prompts.prompt_optimization import (
    MarketCondition,
    batch_build_boosted_prompts,
    build_boosted_prompt,
    get_optimization_engine,
)

logger = logging.getLogger(__name__)


def _sanitize_llm_json(text: str) -> str:
    """Remove markdown, comments, trailing commas"""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Remove comments (// or /* */)
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text.strip()


def extract_scenario_tree(
    llm_output: str, strict: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Extracts the scenario_tree dictionary from LLM output, handling various formats and malformed JSON.
    If strict=True, only accept valid dicts; else, fallback to best-effort extraction.
    Returns a dict if found, else None.
    """

    def try_json_parse(text):
        try:
            sanitized = _sanitize_llm_json(text)
            return json.loads(sanitized)
        except Exception as e:
            print(f"[DEBUG] JSON parse failed: {e}")
            return None

    def extract_from_data(data):
        if not isinstance(data, dict):
            return None
        if "scenario_tree" in data and isinstance(data["scenario_tree"], dict):
            print("[DEBUG] scenario_tree found via direct JSON parse.")
            return data["scenario_tree"]
        if "trades" in data and isinstance(data["trades"], list):
            for trade in data["trades"]:
                if (
                    isinstance(trade, dict)
                    and "scenario_tree" in trade
                    and isinstance(trade["scenario_tree"], dict)
                ):
                    print("[DEBUG] scenario_tree found in trade via JSON parse.")
                    return trade["scenario_tree"]
        return None

    # 1. Try direct JSON parse
    data = try_json_parse(llm_output)
    result = extract_from_data(data) if data else None
    if result:
        return result

    # 2. Try to extract JSON block with scenario_tree
    for match in re.finditer(r"```json(.*?)```", llm_output, re.DOTALL | re.IGNORECASE):
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
            return _extracted_from_extract_scenario_tree_42(
                match, "[DEBUG] scenario_tree found via regex dict pattern."
            )
        except Exception as e:
            print(f"[DEBUG] Regex dict pattern parse failed: {e}")

    # 4. Try to extract from any code block containing scenario_tree
    code_block = re.search(
        r"```[a-zA-Z]*\n(.*?scenario_tree.*?\n.*?)```", llm_output, re.DOTALL
    )
    if code_block:
        block = code_block[1]
        match = re.search(dict_pattern, block, re.DOTALL)
        if match:
            try:
                return _extracted_from_extract_scenario_tree_42(
                    match, "[DEBUG] scenario_tree found in markdown code block."
                )
            except Exception as e:
                print(f"[DEBUG] Markdown code block parse failed: {e}")

    # 5. Fallback: try to extract a dict-like string and parse it
    if not strict:
        dict_like = re.search(r"\{[\s\S]*?\}", llm_output)
        if dict_like:
            try:
                candidate = dict_like[0].replace("'", '"')
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and all(
                    k in parsed for k in ("base_case", "bull_case", "bear_case")
                ):
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


from data_feeds.dark_pools import fetch_dark_pool_data
from data_feeds.earnings_calendar import fetch_earnings_calendar

# 🧩 Import your local scraper modules
from data_feeds.market_internals import fetch_market_internals

# (Legacy) twitter sentiment import retained if needed elsewhere
from data_feeds.news_scraper import fetch_headlines_yahoo_finance
from data_feeds.sources.finviz_scraper import fetch_finviz_breadth
from data_feeds.sources.options_flow import fetch_options_flow
from data_feeds.ticker_universe import fetch_ticker_universe
from oracle_engine.tools import analyze_chart, get_sentiment
from sentiment.sentiment_engine import fetch_sentiment_data
from vector_db.local_store import query_similar
from vector_db.prompt_booster import batch_build_boosted_prompts, build_boosted_prompt

# Ticker validation is optional; fall back to a no-op if the optimizer module
# is unavailable in the runtime environment.
try:
    from optimizations.ticker_validator import validate_tickers
except Exception:  # pragma: no cover - optional dependency

    def validate_tickers(tickers):  # type: ignore
        return tickers


MODEL_NAME = config.model.openai_model


def get_signals_from_scrapers(
    prompt_text: str,
    chart_image_b64: str,
    enable_premium: bool = True,
    optimize: bool = False,
) -> Dict[str, Any]:
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
    internals = (
        fetch_market_internals() if config.data_feeds.fetch_market_internals else {}
    )
    options_flow = (
        fetch_options_flow(tickers) if config.data_feeds.fetch_options_flow else []
    )
    dark_pools = {}
    if config.data_feeds.fetch_dark_pool_data:
        # Dark pool detection with synthetic signal enhancement
        real_dark_pools_dict = fetch_dark_pool_data(tickers)
        real_dark_pools_list = (
            real_dark_pools_dict.get("dark_pools", [])
            if isinstance(real_dark_pools_dict, dict)
            else []
        )

        # Generate synthetic dark pool signals from options flow correlation
        try:
            from data_feeds.synthetic_darkpool_signals import (
                generate_synthetic_darkpool_signals,
            )

            dark_pools_list = generate_synthetic_darkpool_signals(
                options_data=options_flow, real_darkpool_data=real_dark_pools_list
            )
            if dark_pools_list and len(dark_pools_list) > 0:
                print(
                    f"[INFO] Enhanced dark pool detection: {len(dark_pools_list)} signals generated"
                )
                # Return dict in same format as fetch_dark_pool_data
                dark_pools = {
                    "dark_pools": dark_pools_list,
                    "data_source": "synthetic_options_correlation",
                    "timestamp": datetime.now().isoformat(),
                    "total_detected": len(dark_pools_list),
                }
            else:
                dark_pools = real_dark_pools_dict
                print(
                    f"[INFO] Using baseline dark pool detection: {len(real_dark_pools_list)} signals"
                )
        except Exception as e:
            import traceback

            print(f"[WARN] Synthetic dark pool signal generation failed: {e}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            dark_pools = real_dark_pools_dict
    earnings = (
        fetch_earnings_calendar(tickers)
        if config.data_feeds.fetch_earnings_calendar
        else []
    )

    # Sentiment data (Reddit + Twitter - FREE)
    sentiment_web = (
        fetch_sentiment_data(tickers) if config.data_feeds.fetch_sentiment_web else {}
    )

    # News and breadth (web scraping - FREE)
    yahoo_headlines = (
        fetch_headlines_yahoo_finance() if config.data_feeds.fetch_news_yahoo else []
    )
    finviz_breadth = (
        fetch_finviz_breadth() if config.data_feeds.fetch_finviz_breadth else ""
    )

    # LLM-based analysis (local - FREE)
    sentiment_llm = (
        get_sentiment(prompt_text, model_name=MODEL_NAME)
        if config.data_feeds.fetch_sentiment_llm
        else ""
    )

    # Chart analysis (local vision model - FREE)
    import base64

    chart_bytes = None
    if chart_image_b64:
        try:
            chart_bytes = base64.b64decode(chart_image_b64)
        except Exception as e:
            print(f"[DEBUG] Failed to decode chart_image_b64: {e}")
    chart_analysis = (
        analyze_chart(chart_bytes, model_name=MODEL_NAME) if chart_image_b64 else ""
    )

    # Optional: Premium unique data (if API keys available)
    premium_data = None
    if enable_premium and config.data_feeds.enable_premium_feeds:
        try:
            from data_feeds.strategic_premium_feeds import get_premium_feeds

            premium = get_premium_feeds()

            # Only fetch if at least one API is available
            if premium.finnhub_available or premium.fmp_available:
                # Limit to top 5 tickers to avoid rate limits
                top_tickers = tickers[:5]
                print(
                    f"[INFO] Fetching premium unique data for {len(top_tickers)} tickers..."
                )
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

    # Add optimization metadata if enabled
    if optimize:
        try:
            engine = get_optimization_engine()
            market_condition = engine.classify_market_condition(result)
            logger.info(f"Detected market condition: {market_condition.value}")
            result["_market_condition"] = market_condition.value
            result["_optimization_metadata"] = {
                "classification_timestamp": time.time(),
                "signal_count": len(result),
                "prompt_text_length": len(prompt_text),
            }
        except Exception as e:
            logger.warning(f"Optimization metadata failed: {e}")

    return result


def get_signals_from_scrapers_optimized(
    prompt_text: str, chart_image_b64: str
) -> Dict[str, Any]:
    """Wrapper to fetch signals with optimization metadata attached."""
    return get_signals_from_scrapers(
        prompt_text, chart_image_b64, enable_premium=True, optimize=True
    )


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
        payload = hit.get("payload", {})
        if payload:
            text += (
                f"- Ticker: {payload.get('ticker', 'N/A')}, "
                f"Direction: {payload.get('direction', 'N/A')}, "
                f"Thesis: {payload.get('thesis', 'N/A')}, "
                f"Date: {payload.get('date', 'N/A')}\n"
            )
    return text.strip()


def adjust_scenario_tree(
    signals: Dict[str, Any],
    similar_scenarios: str,
    model_name: str = MODEL_NAME,
    optimize: bool = False,
    template_id: Optional[str] = None,
) -> str:
    """
    Calls the LLM to produce an adjusted scenario tree with improved probabilities.
    Returns the LLM's response as a string.
    """
    if optimize:
        content, _ = _adjust_scenario_tree_optimized(
            signals, similar_scenarios, model_name, template_id
        )
        return content

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

    result = dispatch_chat(
        messages=[
            {
                "role": "system",
                "content": "You are ORACLE-X, an adaptive scenario engine.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model_name,
        max_tokens=600,
        temperature=0.3,
        task_type="analytical",
        purpose="adjust_scenario_tree",
        retries=2,
    )
    return result.content


def _adjust_scenario_tree_optimized(
    signals: Dict[str, Any],
    similar_scenarios: str,
    model_name: str = MODEL_NAME,
    template_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced scenario tree adjustment with optimized prompts and performance tracking
    """
    engine = get_optimization_engine()

    # Extract market condition from signals metadata
    market_condition_str = signals.get("_market_condition", "sideways")
    try:
        market_condition = MarketCondition(market_condition_str)
    except ValueError:
        market_condition = MarketCondition.SIDEWAYS

    # Generate optimized prompt
    system_prompt, user_prompt, prompt_metadata = engine.generate_optimized_prompt(
        signals, market_condition, template_id
    )

    logger.info(
        f"Using template: {prompt_metadata['template_id']} for {market_condition.value} market"
    )

    start_time = time.time()
    result = dispatch_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model_name,
        max_tokens=600,
        temperature=0.3,
        task_type="analytical",
        purpose="adjust_scenario_tree_optimized",
        retries=2,
        use_cache=False,
    )
    attempts = [
        {
            "purpose": "adjust_scenario_tree_optimized",
            "model": result.model or model_name,
            "success": bool(result.content),
            "empty": not bool(result.content),
            "error": result.error,
            "latency_sec": round(time.time() - start_time, 4),
        }
    ]

    engine.record_prompt_performance(prompt_metadata, attempts)

    optimization_metadata = {
        "prompt_metadata": prompt_metadata,
        "performance_data": attempts,
        "market_condition": market_condition.value,
    }

    return result.content, optimization_metadata


def adjust_scenario_tree_optimized(
    signals: Dict[str, Any],
    similar_scenarios: str,
    model_name: str = MODEL_NAME,
    template_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Public wrapper for optimized scenario tree generation."""
    return _adjust_scenario_tree_optimized(
        signals, similar_scenarios, model_name, template_id
    )


def adjust_scenario_tree_with_boost(
    signals: Dict[str, Any], similar_scenarios: str, model_name: str = MODEL_NAME
) -> str:
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
    ticker = signals.get("tickers", ["SPY"])[0]
    boosted_prompt = asyncio.run(build_boosted_prompt(base_prompt, ticker))

Analyze and output a scenario tree with updated probabilities for base/bull/bear cases.
Explain how the past scenarios influence your adjustments.
""".strip()

    # Boost the prompt with ChromaDB recall
    boosted_prompt = build_boosted_prompt(
        base_prompt, str(signals.get("chart_analysis", ""))
    )
    result = dispatch_chat(
        messages=[
            {
                "role": "system",
                "content": "You are ORACLE-X, an adaptive scenario engine.",
            },
            {"role": "user", "content": boosted_prompt},
        ],
        model=model_name,
        max_tokens=600,
        temperature=0.3,
        task_type="analytical",
        purpose="adjust_scenario_tree_with_boost",
        retries=2,
        use_cache=False,
    )
    return result.content


def batch_adjust_scenario_trees_with_boost(
    signals_list: List[Dict[str, Any]],
    similar_scenarios_list: List[str],
    model_name: str = MODEL_NAME,
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
    tickers = [s.get("tickers", ["SPY"])[0] for s in signals_list]
    boosted_prompts = asyncio.run(batch_build_boosted_prompts(base_prompts, tickers))

Analyze and output a scenario tree with updated probabilities for base/bull/bear cases.
Explain how the past scenarios influence your adjustments.
""".strip()
        base_prompts.append(base_prompt)
    # Use batch prompt boosting
    trade_theses = [str(s.get("chart_analysis", "")) for s in signals_list]
    boosted_prompts = batch_build_boosted_prompts(base_prompts, trade_theses)
    results = []
    for boosted_prompt in boosted_prompts:
        dispatch_result = dispatch_chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are ORACLE-X, an adaptive scenario engine.",
                },
                {"role": "user", "content": boosted_prompt},
            ],
            model=model_name,
            max_tokens=600,
            temperature=0.3,
            task_type="analytical",
            purpose="batch_adjust_scenario_trees_with_boost",
            retries=2,
            use_cache=False,
        )
        results.append(dispatch_result.content)
    return results


def generate_final_playbook(
    signals,
    scenario_tree,
    model_name=MODEL_NAME,
    optimize: bool = False,
    template_id: Optional[str] = None,
):
    """
    Final LLM step: generates tomorrow's best trades + 'Tomorrow's Tape'.
    """
    if optimize:
        content, _ = _generate_final_playbook_optimized(
            signals, scenario_tree, model_name, template_id
        )
        return content

    # Clean signals before prompt construction
    signals = clean_signals_for_llm(signals)

    # Get real-time prices for ALL tickers using orchestrator data
    current_prices = {}
    all_tickers = set()

    # Extract all tickers from signals
    if "tickers" in signals and signals["tickers"]:
        all_tickers.update(signals["tickers"])

    # Extract tickers from scenario tree
    if scenario_tree:
        import re

        scenario_tickers = re.findall(r"\b[A-Z]{2,5}\b", scenario_tree)
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
                if quote and hasattr(quote, "price") and quote.price:
                    current_prices[ticker] = f"{float(quote.price):.2f}"
                else:
                    # Fallback to market data
                    market_data = orch.get_market_data(
                        ticker, period="1d", interval="1d"
                    )
                    if (
                        market_data
                        and hasattr(market_data, "data")
                        and not market_data.data.empty
                    ):
                        latest_close = market_data.data["Close"].iloc[-1]
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

    result = dispatch_chat(
        messages=[
            {
                "role": "system",
                "content": "You are ORACLE-X, the final Playbook composer.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model_name,
        max_tokens=1024,
        temperature=0.3,
        task_type="analytical",
        purpose="generate_final_playbook",
        retries=2,
        use_cache=False,
    )
    return result.content


def _generate_final_playbook_optimized(
    signals: Dict[str, Any],
    scenario_tree: str,
    model_name: str = MODEL_NAME,
    template_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced playbook generation with optimized prompts and comprehensive tracking
    """
    engine = get_optimization_engine()

    # Clean signals first
    cleaned_signals = clean_signals_for_llm(signals)

    # Extract market condition
    market_condition_str = signals.get("_market_condition", "sideways")
    try:
        market_condition = MarketCondition(market_condition_str)
    except ValueError:
        market_condition = MarketCondition.SIDEWAYS

    # For playbook generation, prefer templates that focus on trading execution
    if template_id is None:
        # Select template optimized for final trading decisions
        template_preferences = [
            ("conservative_balanced", MarketCondition.SIDEWAYS),
            ("aggressive_momentum", MarketCondition.BULLISH),
            ("aggressive_momentum", MarketCondition.VOLATILE),
            ("earnings_specialist", MarketCondition.EARNINGS),
            ("technical_precision", MarketCondition.BEARISH),
        ]

        for temp_id, condition in template_preferences:
            if condition == market_condition:
                template_id = temp_id
                break

        if template_id is None:
            template_id = "conservative_balanced"  # Safe default

    # Generate optimized prompt for playbook generation
    system_prompt, user_prompt_base, prompt_metadata = engine.generate_optimized_prompt(
        cleaned_signals, market_condition, template_id
    )

    # Get real-time prices for ALL tickers using orchestrator data
    current_prices = {}
    all_tickers = set()

    # Extract all tickers from signals
    if "tickers" in cleaned_signals and cleaned_signals["tickers"]:
        all_tickers.update(cleaned_signals["tickers"])

    # Extract tickers from scenario tree
    if scenario_tree:
        import re

        scenario_tickers = re.findall(r"\b[A-Z]{2,5}\b", scenario_tree)
        all_tickers.update([t for t in scenario_tickers if len(t) <= 5 and t.isalpha()])

    # Validate the merged ticker list
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
                quote = orch.get_quote(ticker)
                if quote and hasattr(quote, "price") and quote.price:
                    current_prices[ticker] = f"{float(quote.price):.2f}"
                else:
                    market_data = orch.get_market_data(
                        ticker, period="1d", interval="1d"
                    )
                    if (
                        market_data
                        and hasattr(market_data, "data")
                        and not market_data.data.empty
                    ):
                        latest_close = market_data.data["Close"].iloc[-1]
                        current_prices[ticker] = f"{float(latest_close):.2f}"
            except Exception as e:
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

    # Enhance prompt with scenario tree context
    enhanced_user_prompt = f"""
{user_prompt_base}

{price_context}
Adjusted Scenario Tree:
{scenario_tree}

Based on this analysis, generate the 1–3 highest-confidence trades for tomorrow.
Include: ticker, direction, instrument, entry range, profit target, stop-loss,
counter-signal, and a 5-sentence 'Tomorrow's Tape'.

Format your response as **valid JSON only** (no markdown, no extra text, no comments) with:
- 'trades': a list of trade objects
- 'tomorrows_tape': a string summary

IMPORTANT: For each trade, include a 'thesis' field summarizing the core rationale in 1-2 sentences.
IMPORTANT: For each trade, include a 'scenario_tree' field with a dictionary of scenario probabilities (base_case, bull_case, bear_case)

DO NOT include any text before or after the JSON. Output only the JSON object.
"""

    # Execute LLM call
    start_time = time.time()
    result = dispatch_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_user_prompt},
        ],
        model=model_name,
        max_tokens=1024,
        temperature=0.3,
        task_type="analytical",
        purpose="generate_final_playbook_optimized",
        retries=2,
        use_cache=False,
    )
    content = result.content
    attempts = [
        {
            "purpose": "generate_final_playbook_optimized",
            "model": result.model or model_name,
            "success": bool(result.content),
            "empty": not bool(result.content),
            "error": result.error,
            "latency_sec": round(time.time() - start_time, 4),
        }
    ]

    # Record performance
    engine.record_prompt_performance(prompt_metadata, attempts)

    # Analyze output quality
    output_quality = analyze_playbook_quality(content)

    optimization_metadata = {
        "prompt_metadata": prompt_metadata,
        "performance_data": attempts,
        "market_condition": market_condition.value,
        "output_quality": output_quality,
        "template_used": template_id,
    }

    return content, optimization_metadata


def generate_final_playbook_optimized(
    signals: Dict[str, Any],
    scenario_tree: str,
    model_name: str = MODEL_NAME,
    template_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Public wrapper for optimized playbook generation."""
    return _generate_final_playbook_optimized(
        signals, scenario_tree, model_name, template_id
    )


def analyze_playbook_quality(playbook_json: str) -> Dict[str, Any]:
    """
    Analyze the quality of generated playbook for learning purposes
    """
    quality_metrics = {
        "valid_json": False,
        "has_trades": False,
        "has_tomorrows_tape": False,
        "trade_count": 0,
        "trades_have_required_fields": False,
        "confidence_score": 0.0,
        "completeness_score": 0.0,
    }

    try:
        # Parse JSON
        data = json.loads(playbook_json)
        quality_metrics["valid_json"] = True

        # Check structure
        if "trades" in data and isinstance(data["trades"], list):
            quality_metrics["has_trades"] = True
            quality_metrics["trade_count"] = len(data["trades"])

            # Check trade fields
            required_fields = [
                "ticker",
                "direction",
                "entry_range",
                "profit_target",
                "stop_loss",
            ]
            if data["trades"]:
                first_trade = data["trades"][0]
                has_required = all(field in first_trade for field in required_fields)
                quality_metrics["trades_have_required_fields"] = has_required

        if (
            "tomorrows_tape" in data
            and isinstance(data["tomorrows_tape"], str)
            and data["tomorrows_tape"].strip()
        ):
            quality_metrics["has_tomorrows_tape"] = True

        # Calculate overall scores
        structure_score = (
            sum(
                [
                    quality_metrics["valid_json"],
                    quality_metrics["has_trades"],
                    quality_metrics["has_tomorrows_tape"],
                    quality_metrics["trades_have_required_fields"],
                ]
            )
            / 4
        )

        content_score = min(
            1.0, quality_metrics["trade_count"] / 2
        )  # Optimal is 1-2 trades

        quality_metrics["completeness_score"] = (structure_score + content_score) / 2
        quality_metrics["confidence_score"] = quality_metrics["completeness_score"]

    except json.JSONDecodeError:
        quality_metrics["valid_json"] = False
    except Exception as e:
        logger.warning(f"Error analyzing playbook quality: {e}")

    return quality_metrics


def get_optimization_analytics() -> Dict[str, Any]:
    """Get comprehensive optimization analytics"""
    engine = get_optimization_engine()
    return engine.get_performance_analytics()


def evolve_prompt_templates(performance_threshold: float = 0.7) -> List[str]:
    """Evolve prompt templates based on performance data"""
    engine = get_optimization_engine()
    analytics = engine.get_performance_analytics()

    # Identify top performers
    top_performers = []
    for template_perf in analytics.get("template_performance", []):
        if template_perf["avg_success_rate"] >= performance_threshold:
            top_performers.append(template_perf["template_id"])

    if not top_performers:
        logger.warning("No top performing templates found for evolution")
        return []

    # Evolve templates
    new_templates = engine.evolve_prompts(top_performers)
    evolved_ids = []

    for template in new_templates:
        engine.prompt_templates[template.template_id] = template
        evolved_ids.append(template.template_id)
        logger.info(f"Created evolved template: {template.template_id}")

    return evolved_ids


def run_optimization_experiment(
    signals: Dict[str, Any], duration_minutes: int = 30
) -> Dict[str, Any]:
    """
    Run an optimization experiment comparing different prompt strategies
    """
    engine = get_optimization_engine()
    market_condition = engine.classify_market_condition(signals)

    # Select templates to test
    test_templates = [
        "conservative_balanced",
        "aggressive_momentum",
        "technical_precision",
    ]
    available_templates = [
        tid for tid in test_templates if tid in engine.prompt_templates
    ]

    if len(available_templates) < 2:
        logger.warning("Not enough templates available for experiment")
        return {}

    # Run A/B test
    template_a, template_b = available_templates[0], available_templates[1]
    experiment_id = engine.start_ab_test(
        template_a, template_b, market_condition, duration_minutes // 60
    )

    logger.info(
        f"Started optimization experiment {experiment_id}: {template_a} vs {template_b}"
    )

    results = {}
    for template_id in [template_a, template_b]:
        system_prompt, user_prompt, metadata = engine.generate_optimized_prompt(
            signals, market_condition, template_id
        )

        # Generate sample outputs
        dispatch_result = dispatch_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=MODEL_NAME,
            max_tokens=600,
            temperature=0.3,
            task_type="analytical",
            purpose="run_optimization_experiment",
            retries=2,
            use_cache=False,
        )
        quality = analyze_playbook_quality(dispatch_result.content)

        results[template_id] = {
            "metadata": metadata,
            "output_quality": quality,
            "content_length": len(dispatch_result.content),
            "success": len(dispatch_result.content) > 0,
            "error": dispatch_result.error,
        }

    return {
        "experiment_id": experiment_id,
        "market_condition": market_condition.value,
        "results": results,
        "recommendation": _analyze_experiment_results(results),
    }


def _analyze_experiment_results(results: Dict[str, Any]) -> str:
    """Analyze experiment results and provide recommendation"""
    if len(results) < 2:
        return "Insufficient data for analysis"

    template_scores = {}
    for template_id, result in results.items():
        if result.get("success"):
            quality = result.get("output_quality", {})
            score = quality.get("completeness_score", 0)
            template_scores[template_id] = score
        else:
            template_scores[template_id] = 0

    if not template_scores:
        return "All templates failed"

    best_template = max(template_scores.items(), key=lambda x: x[1])
    return f"Recommended template: {best_template[0]} (score: {best_template[1]:.2f})"
