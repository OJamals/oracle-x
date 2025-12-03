# ORACLE-X Copilot Instructions: Refactored Patterns

Post-P1-P5 refactor, follow these patterns for AI agents, prompt chains, and oracle_engine integrations. Prioritize unified interfaces for maintainability.

## 🎯 Core Principles
- **Route through Orchestrators**: Never call adapters directly; use [`data_feeds.data_feed_orchestrator`](data_feeds/data_feed_orchestrator.py) or submodules.
- **Cache Everything**: Mandatory via [`UnifiedCacheManager`](core/cache/unified_cache_manager.py).
- **Sentiment First**: Aggregate before LLM prompts.
- **Specific Exceptions**: Avoid bare `except Exception`; use/raise typed errors.
- **Async Where Possible**: Leverage orchestrator's async fallbacks.

## 1. Data Feed Orchestration
**Main Entry**: [`data_feeds.data_feed_orchestrator.DataFeedOrchestrator`](data_feeds/data_feed_orchestrator.py)

```python
# Preferred: Factory function
from data_feeds.data_feed_orchestrator import get_orchestrator, get_quote, get_sentiment_data

orchestrator = get_orchestrator()
quote = get_quote("AAPL")  # Auto-fallback, cache
market_data = get_market_data("AAPL", period="1y")
sentiment = get_sentiment_data("AAPL")  # Unified multi-source

# Advanced
advanced_sentiment = orchestrator.get_advanced_sentiment_data("AAPL")
system_health = orchestrator.get_system_health()
```

**Submodules Usage** (post-split):
```python
# Validation
from data_feeds.orchestrator.validation.data_validator import validate_data
validated = validate_data(raw_data)

# Utils
from data_feeds.orchestrator.utils.helpers import format_ticker_data
from data_feeds.orchestrator.utils.performance_tracker import track_perf
```

## 2. Unified Caching
**Single Interface**: [`core/cache/unified_cache_manager.UnifiedCacheManager`](core/cache/unified_cache_manager.py)

```python
from core.cache.unified_cache_manager import UnifiedCacheManager

cache = UnifiedCacheManager.get_instance(ttl=300)  # 5min TTL

# Get with compute
def fetch_sentiment(ticker):
    return orchestrator.get_sentiment_data(ticker)

sentiment = cache.get(f"sentiment:{ticker}", fetch_sentiment)
cache.set("key", value, ttl=600)
cache.invalidate_pattern("sentiment:AAPL*")
```

**Best Practice**: Wrap all external calls (API, LLM).

## 3. Sentiment Engine
**Unified**: [`sentiment/sentiment_engine.SentimentEngine`](sentiment/sentiment_engine.py) or `AdvancedSentimentEngine`

```python
from sentiment.sentiment_engine import AdvancedSentimentEngine

engine = AdvancedSentimentEngine()
result = engine.get_sentiment_data(
    symbol="AAPL",
    sources=["reddit", "twitter", "news"],
    max_per_source=200
)
# Returns: dict with scores, raw_data['aggregated_counts']
```

**Integration**: Call before `prompt_chain.py`; inject `result['composite_score']` into prompts.

## 4. Structured Error Handling
**Pattern**: Specific exceptions over generic.

```python
# Example (adapt to module)
try:
    data = orchestrator.get_quote("INVALID")
except ValueError as e:  # Invalid ticker
    logger.warning(f"Ticker validation failed: {e}")
    use_fallback()
except ConnectionError as e:  # API down
    logger.error(f"Feed unavailable: {e}")
    orchestrator.fallback_mode()
except Exception as e:  # Last resort
    logger.critical(f"Unexpected: {e}")
    raise
```

**Copilot Tip**: Generate `raise DataFeedError("msg")` for new errors; log + fallback.

## 5. Oracle Engine / Prompt Chain Integration
**Example in [`oracle_engine/chains/prompt_chain.py`](oracle_engine/chains/prompt_chain.py)**:

```python
def generate_playbook(symbol):
    cache = UnifiedCacheManager.get_instance()

    # Data
    data = cache.get(f"playbook_data:{symbol}", lambda: {
        "quote": get_quote(symbol),
        "sentiment": get_sentiment_data(symbol),
        "health": get_system_health()
    })

    # Prompt with sentiment bias
    prompt = f"Generate playbook for {symbol}. Sentiment: {data['sentiment']['composite_score']:.2f}"

    response = llm_client.generate(prompt)
    return response
```

## 6. Performance & Validation
- **Track**: `performance_tracker.track("sentiment_fetch", duration)`
- **Validate**: `data_validator.validate(data, schema="quote")`
- **Health**: Check `orchestrator.get_system_health()` before heavy ops.

## 🚫 Avoid
- Direct adapter calls (e.g., `GrokTwitterAdapter`)
- Multiple cache impls
- Bare `except:`
- Uncached API/LLM calls

**See**: [`refactoring_plan.md`](refactoring_plan.md) for migration details.

**Updated**: Post-P1-P5 complete.
