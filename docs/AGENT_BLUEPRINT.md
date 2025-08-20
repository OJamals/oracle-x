# Autonomous LLM Agent Blueprint over the DataFeedOrchestrator

Purpose: enable an LLM agent to autonomously query orchestrator endpoints and synthesize decision-grade answers to complex finance questions, e.g., “What are the best US stock options to buy on Aug 6, with highest chance of appreciating due to company momentum, underpricing, or triggers”.

---

## 1) Architecture Overview

1. Tools Layer (function-calling over orchestrator)
   - Wrap orchestrator endpoints as typed tools with strict schemas:
     - [`data_feeds.data_feed_orchestrator.get_quote()`](data_feeds/data_feed_orchestrator.py:1449)
     - [`data_feeds.data_feed_orchestrator.get_market_data()`](data_feeds/data_feed_orchestrator.py:1453)
     - [`data_feeds.data_feed_orchestrator.get_sentiment_data()`](data_feeds/data_feed_orchestrator.py:1457)
     - [`data_feeds.data_feed_orchestrator.get_advanced_sentiment()`](data_feeds/data_feed_orchestrator.py:1461)
     - [`data_feeds.data_feed_orchestrator.get_finviz_news()`](data_feeds/data_feed_orchestrator.py:1477)
     - [`data_feeds.data_feed_orchestrator.get_market_breadth()`](data_feeds/data_feed_orchestrator.py:1469)
     - [`data_feeds.data_feed_orchestrator.get_sector_performance()`](data_feeds/data_feed_orchestrator.py:1473)
     - [`data_feeds.data_feed_orchestrator.get_company_info()`](data_feeds/data_feed_orchestrator.py:1421)
     - [`data_feeds.data_feed_orchestrator.get_financial_statements()`](data_feeds/data_feed_orchestrator.py:1431)
     - [`data_feeds.data_feed_orchestrator.get_multiple_quotes()`](data_feeds/data_feed_orchestrator.py:1427)
   - Optional future tool: options_screener(symbols, risk_profile, budget) using an options chain source.

2. Reasoning & Planning Layer
   - Agent converts user query → plan → executes tools → aggregates → ranks → explains.
   - Plans respect rate limits and caching in orchestrator (SmartCache, RateLimiter).

3. Scoring & Ranking Layer
   - Compute a transparent composite score S for each candidate:
     S = w_m * Momentum + w_v * ValuationUndervalued + w_t * TriggerLikelihood + w_s * SentimentStrength
   - Normalize each component to [0,1]. Provide per-component diagnostics for explainability.

4. Output Layer
   - Final ranked JSON with:
     - top picks, scores, per-factor breakdown, confidence, supporting evidence (headlines, sentiment summary, returns).

---

## 2) Tools Layer (Wrappers)

Create a wrapper module exposing strict, JSON-serializable outputs. Example sketch:

- File: `data_feeds_agent_tools.py`
- Uses [`data_feeds.data_feed_orchestrator.get_orchestrator()`](data_feeds/data_feed_orchestrator.py:1442)

Functions (examples):
- get_quote_tool(symbol: str) → dict
- get_market_data_tool(symbol: str, period="6mo", interval="1d") → dict with minimal OHLCV-derived features (1w/1m returns)
- get_sentiment_tool(symbol: str) → dict (Reddit/Twitter)
- get_advanced_sentiment_tool(symbol: str) → dict (ensemble score, confidence, sample_size)
- get_news_tool(limit: int=50) → dict (FinViz + Yahoo Finance)
- get_breadth_tool() → dict
- get_sector_perf_tool() → list[dict]
- get_company_info_tool(symbol: str) → dict
- get_financials_tool(symbol: str) → dict of frames/metrics (optional normalization)

All wrappers must:
- Return only JSON-safe types
- Handle None and empty cases gracefully
- Batch where possible (e.g., get_multiple_quotes)

---

## 3) Candidate Generation

Aim: build a reasonable symbol list before deep analysis:
- Sector performance leaders: [`data_feeds.data_feed_orchestrator.get_sector_performance()`](data_feeds/data_feed_orchestrator.py:1473)
- Market breadth context: [`data_feeds.data_feed_orchestrator.get_market_breadth()`](data_feeds/data_feed_orchestrator.py:1469)
- Optional: user-provided watchlist, or pre-built ticker universe from [`data_feeds.ticker_universe`](data_feeds/ticker_universe.py:1)

Heuristics:
- Prefer top sectors by recent performance
- Include sector leaders (large cap/liquid)
- Limit initial universe to manageable N (e.g., 100–200) to respect rate limits

---

## 4) Features and Scoring

Compute factors per symbol:

1) Momentum (0–1)
- From market data: 1w and 1m returns, breakouts (e.g., Close > recent high)
- Normalize via logistic/sigmoid or percentile rank

2) ValuationUndervalued (0–1)
- If fundamentals available: PE vs sector, PEG, FCF yield
- Otherwise leave as neutral until a fundamental source is integrated

3) TriggerLikelihood (0–1)
- Headline intensity from FinViz + Yahoo: counts and recency
- Proximity to earnings date from company info (if available)
- Insider activity, notable events (as available)

4) SentimentStrength (0–1)
- Use advanced ensemble sentiment:
  - [`data_feeds.advanced_sentiment.AdvancedSentimentEngine`](data_feeds/advanced_sentiment.py:278)
  - Aggregates Reddit/Twitter/News texts in orchestrator with caps, truncation, and dedupe:
    - [`data_feeds.data_feed_orchestrator.get_advanced_sentiment_data()`](data_feeds/data_feed_orchestrator.py:1149)
- Convert to positive strength = max(0, ensemble_score) * confidence

Composite Score:
- S = w_m*M + w_v*V + w_t*T + w_s*Snt; choose weights reflecting your strategy (e.g., 0.35/0.20/0.20/0.25)
- Provide all component values in final output for transparency.

Confidence:
- Derive from:
  - Sentiment confidence
  - Data completeness (market data availability, news coverage)
  - Dispersion checks (variance across models, implemented in [`AdvancedSentimentEngine.analyze_text()`](data_feeds/advanced_sentiment.py:307))

Risk Controls:
- Liquidity: require minimum ADV
- Sector diversification: limit per sector
- Event awareness: treat pre-earnings risk separately

---

## 5) Agent Loop

Workflow:
1. Parse Intent
   - Extract constraints: date, “best US stock options”, themes (momentum, underpricing, triggers)

2. Plan
   - Build candidate list
   - Decide which tools to call for each stage (market data, sentiment, news, company info)
   - Budget tool calls (max depth, batch where possible)

3. Execute Tools
   - Use orchestrator wrappers; they internally use caching & rate limiting

4. Aggregate & Score
   - Compute factor scores and composite score
   - Rank symbols

5. Explain
   - For each top pick: show factor breakdown, recent returns, key headlines, sentiment summary

6. Output
   - JSON suitable for UI or trading pipeline
   - Optional: produce a textual recommendation

---

## 6) CLI Entry Point

Add a command to run the agent end-to-end:
- Example:
  ```
  python main.py agent --query "best US stock options to buy on 2025-08-06 with momentum/undervaluation/triggers" --max-symbols 120 --top 10
  ```
- The agent:
  - Builds candidates
  - Gathers features via tools
  - Ranks and prints a structured JSON + human-readable summary

---

## 7) Options Suggestions (Future)

If/when options chain data is integrated:
- For momentum picks:
  - Buy calls or bull call spreads around 30–45 DTE, delta ~0.25–0.40
- For underpricing with catalysts:
  - Consider calendars or verticals depending on IV regime
- Add a tool `recommend_options(symbols, risk_profile, budget)` to compute candidates

---

## 8) Safety, Caching, and Limits

- Rely on orchestrator SmartCache and RateLimiter (already implemented in [`data_feeds.data_feed_orchestrator`](data_feeds/data_feed_orchestrator.py:333))
- Batch calls where possible (e.g., get_multiple_quotes)
- Environment toggles for debug logging (e.g., DEBUG_NEWS_AGG=1)

---

## 9) Implementation Files and Key References

- Orchestrator endpoints and adapters:
  - [`data_feeds.data_feed_orchestrator.DataFeedOrchestrator`](data_feeds/data_feed_orchestrator.py:820)
  - Reddit sentiment adapter: [`data_feeds.data_feed_orchestrator.RedditAdapter`](data_feeds/data_feed_orchestrator.py:525)
  - Twitter sentiment adapter: [`data_feeds.data_feed_orchestrator.TwitterAdapter`](data_feeds/data_feed_orchestrator.py:654)
  - Advanced sentiment adapter: [`data_feeds.data_feed_orchestrator.AdvancedSentimentAdapter`](data_feeds/data_feed_orchestrator.py:747)
  - FinViz news integration (used for headlines aggregation): [`data_feeds.data_feed_orchestrator.get_advanced_sentiment_data()`](data_feeds/data_feed_orchestrator.py:1196)
  - Yahoo Finance headlines: [`data_feeds.news_scraper.fetch_headlines_yahoo_finance()`](data_feeds/news_scraper.py:7)

- Advanced sentiment engine:
  - [`data_feeds.advanced_sentiment.AdvancedSentimentEngine`](data_feeds/advanced_sentiment.py:278)
  - FinBERT concurrency-safe inference: [`data_feeds.advanced_sentiment.FinBERTAnalyzer.analyze()`](data_feeds/advanced_sentiment.py:221)
  - Financial lexicon with contextual qualifiers: [`data_feeds.advanced_sentiment.FinancialLexicon`](data_feeds/advanced_sentiment.py:59)

---

## 10) Phased Plan

Phase 1 (MVP):
- Implement tools wrappers (data_feeds_agent_tools.py)
- Agent class with plan → gather → score → explain
- CLI entrypoint to execute

Phase 2 (Enhancements):
- Add valuation factors if fundamentals available
- Integrate options chain data and options recommender
- Improve candidate generation with sector and breadth leaders

Phase 3 (Production Polish):
- Add persistent logs and telemetry for plans and tool calls
- Add unit/integration tests for agent workflows
- Tighten output schemas for UI/trading pipelines

---

## 11) Guiding Principles

- No placeholders/mocks. Only live, implemented sources (Reddit, Twitter, Yahoo, FinViz, market data).
- Deterministic, explainable scoring with component breakdowns.
- Respect rate limits and use cache.
- Defensive programming: handle None/empty paths gracefully.