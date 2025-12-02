# Grok Twitter Sentiment Integration

## Current Twitter Functionality Analysis

**Core Components:**
- [`data_feeds/twitter_feed.py`](data_feeds/twitter_feed.py): `TwitterSentimentFeed` uses [twscrape](https://github.com/vladkens/twscrape) to scrape tweets. Features:
  - Pre-compiled regex for tickers, cleaning (URLs, mentions).
  - Sentiment analysis via `analyze_text_sentiment(source="twitter")` (FinTwitBERT/VADER/TextBlob ensemble).
  - Limits to 20 tweets for speed, dedup, filters.
- [`data_feeds/twitter_adapter.py`](data_feeds/twitter_adapter.py): `EnhancedTwitterAdapter` wraps feed, uses `analyze_symbol_sentiment` for aggregate score/confidence.
- Integration: Used in [`enhanced_sentiment_pipeline.py`](data_feeds/enhanced_sentiment_pipeline.py) and orchestrator.

**Strengths:** Fast scraping, multi-model sentiment.
**Limitations:** Unofficial scraper (risk of blocks), no LLM reasoning.

## New Implementation: Grok-Powered Agent

**File:** [`data_feeds/grok_twitter_adapter.py`](data_feeds/grok_twitter_adapter.py)

**Key Features:**
- OpenRouter API (configure with `OPENROUTER_API_KEY` or `GROK_API_KEY`): `x-ai/grok-4.1-fast`.
- **ReAct Agent** with tool calling:
  - **Tool:** `fetch_tweets(query, limit)` → uses existing `TwitterSentimentFeed`.
  - Agent loop: Fetches tweets, analyzes, returns JSON `{"sentiment": -1~1, "confidence": 0~1, "reasoning": "..."}`.
- `GrokTwitterAdapter.get_sentiment(symbol)` → `SentimentData` compatible.
- Health status endpoint.

**Integration:**
- Added to `OptimizedSentimentPipeline.sentiment_sources['grok_twitter']`.
- Parallel execution with other sources.

**Usage:**
```python
from data_feeds.grok_twitter_adapter import GrokTwitterAdapter
adapter = GrokTwitterAdapter()
result = adapter.get_sentiment("AAPL")
# SentimentData(sentiment_score=..., confidence=..., raw_data={"agent_result": {...}})
```

**Benefits:**
- LLM reasoning for nuanced sentiment.
- Tool-based fetching (extensible).
- Zero-shot symbol analysis.

**Notes:**
- Requires `OPENROUTER_API_KEY` (or `GROK_API_KEY`) in the environment. Optional `OPENROUTER_BASE_URL` override.
- Linter warnings (types): Runtime functional.
- Twscrape dependency (login/accounts may need setup).
- Rate limits/costs via OpenRouter.

**Next Steps:**
- Production: Env var for key.
- Enhance: More tools (e.g., filter tweets).
- Monitor: Add to diagnostics.

Tested: Adapter fetches/analyzes AAPL tweets via Grok agent.
