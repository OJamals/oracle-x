# Oracle-X Data Orchestrator – Endpoint Catalog

This catalog enumerates all data endpoints exposed by the unified DataFeedOrchestrator, its module-level convenience functions, and the compatibility layers that delegate to the legacy ConsolidatedDataFeed. Duplicate coverage across providers is noted.

Source summary:
- Orchestrator: [data_feeds.data_feed_orchestrator.get_orchestrator()](data_feeds/data_feed_orchestrator.py:1549)
- Consolidated feed: [data_feeds.consolidated_data_feed.ConsolidatedDataFeed](data_feeds/consolidated_data_feed.py:909)
- Unified facade: [data_feeds.data_feeds_unified.UnifiedDataProvider](data_feeds/data_feeds_unified.py:42)
- Oracle interface: [data_feeds.oracle_data_interface.OracleDataProvider](data_feeds/oracle_data_interface.py:39)

## 1) Orchestrator Class Endpoints

Class: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator](data_feeds/data_feed_orchestrator.py:820)

Adapters initialized (sources):
- YFinance (quotes, historical): [data_feeds.data_feed_orchestrator.YFinanceAdapter](data_feeds/data_feed_orchestrator.py:424)
- Reddit sentiment: [data_feeds.data_feed_orchestrator.RedditAdapter](data_feeds/data_feed_orchestrator.py:525)
- Twitter sentiment: [data_feeds.data_feed_orchestrator.TwitterAdapter](data_feeds/data_feed_orchestrator.py:654)
- TwelveData: [data_feeds.twelvedata_adapter.TwelveDataAdapter](data_feeds/data_feed_orchestrator.py:857)
- FinViz: [data_feeds.finviz_adapter.FinVizAdapter](data_feeds/data_feed_orchestrator.py:858)
- Advanced sentiment aggregator: [data_feeds.data_feed_orchestrator.AdvancedSentimentAdapter](data_feeds/data_feed_orchestrator.py:747)

Endpoints:
1. Quotes
   - get_quote(symbol, preferred_sources=None)
     - Source(s): yfinance wrapper and adapter; optionally TwelveData, Finnhub via standardized wrappers when available
     - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_quote()](data_feeds/data_feed_orchestrator.py:881)
     - Output: Quote dataclass with quality score

2. Historical Market Data
   - get_market_data(symbol, period="1y", interval="1d", preferred_sources=None)
     - Source(s): yfinance primarily, optionally wrappers (finnhub/fmp/finance_db if wrappers provide)
     - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_market_data()](data_feeds/data_feed_orchestrator.py:967)
     - Output: MarketData dataclass

3. Company Info (delegated)
   - get_company_info(symbol)
     - Delegates to ConsolidatedDataFeed
     - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_company_info()](data_feeds/data_feed_orchestrator.py:1062)

4. Company News (delegated)
   - get_news(symbol, limit=10)
     - Delegates to ConsolidatedDataFeed
     - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_news()](data_feeds/data_feed_orchestrator.py:1079)

5. Multiple Quotes (delegated)
   - get_multiple_quotes(symbols)
     - Delegates to ConsolidatedDataFeed
     - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_multiple_quotes()](data_feeds/data_feed_orchestrator.py:1095)

6. Financial Statements (delegated)
   - get_financial_statements(symbol)
     - Delegates to ConsolidatedDataFeed
     - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_financial_statements()](data_feeds/data_feed_orchestrator.py:1108)

7. Sentiment – per-source
   - get_sentiment_data(symbol, sources=None=[Reddit, Twitter])
     - Source(s): Reddit (batch cache via [data_feeds.reddit_sentiment.fetch_reddit_sentiment()](data_feeds/data_feed_orchestrator.py:560)), Twitter via TwitterSentimentFeed
     - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_sentiment_data()](data_feeds/data_feed_orchestrator.py:1121)
     - Output: map source → SentimentData

8. Advanced Sentiment – multi-source text aggregation
   - get_advanced_sentiment_data(symbol, texts=None, sources=None)
     - Aggregates: Reddit sample_texts, Twitter tweets, and News (Yahoo Finance headlines via [data_feeds.news_scraper.fetch_headlines_yahoo_finance()](data_feeds/data_feed_orchestrator.py:1201) and FinViz headlines via FinViz adapter)
     - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_advanced_sentiment_data()](data_feeds/data_feed_orchestrator.py:1149)
     - Output: SentimentData

9. Market Breadth (FinViz)
   - get_market_breadth()
     - Source: FinViz adapter
     - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_market_breadth()](data_feeds/data_feed_orchestrator.py:1341)

10. Sector Performance (FinViz)
    - get_sector_performance()
      - Source: FinViz adapter
      - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_sector_performance()](data_feeds/data_feed_orchestrator.py:1370)

11. FinViz Collections
    - get_finviz_news()
      - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_news()](data_feeds/data_feed_orchestrator.py:1399)
    - get_finviz_insider_trading()
      - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_insider_trading()](data_feeds/data_feed_orchestrator.py:1423)
    - get_finviz_earnings()
      - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_earnings()](data_feeds/data_feed_orchestrator.py:1448)
    - get_finviz_forex()
      - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_forex()](data_feeds/data_feed_orchestrator.py:1473)
    - get_finviz_crypto()
      - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_crypto()](data_feeds/data_feed_orchestrator.py:1498)

12. Quality / System Health
    - get_data_quality_report()
      - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_data_quality_report()](data_feeds/data_feed_orchestrator.py:1301)
    - validate_system_health()
      - Ref: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.validate_system_health()](data_feeds/data_feed_orchestrator.py:1320)

## 2) Orchestrator Module-Level Convenience Endpoints

Module functions forwarding to a singleton orchestrator:
- get_quote(symbol)
  - Ref: [data_feeds.data_feed_orchestrator.get_quote()](data_feeds/data_feed_orchestrator.py:1556)
- get_market_data(symbol, period="1y", interval="1d")
  - Ref: [data_feeds.data_feed_orchestrator.get_market_data()](data_feeds/data_feed_orchestrator.py:1560)
- get_sentiment_data(symbol)
  - Ref: [data_feeds.data_feed_orchestrator.get_sentiment_data()](data_feeds/data_feed_orchestrator.py:1564)
- get_advanced_sentiment(symbol, texts=None, sources=None)
  - Ref: [data_feeds.data_feed_orchestrator.get_advanced_sentiment()](data_feeds/data_feed_orchestrator.py:1568)
- get_system_health()
  - Ref: [data_feeds.data_feed_orchestrator.get_system_health()](data_feeds/data_feed_orchestrator.py:1572)
- get_company_info(symbol)
  - Ref: [data_feeds.data_feed_orchestrator.get_company_info()](data_feeds/data_feed_orchestrator.py:1526)
- get_news(symbol, limit=10)
  - Ref: [data_feeds.data_feed_orchestrator.get_news()](data_feeds/data_feed_orchestrator.py:1530)
- get_multiple_quotes(symbols)
  - Ref: [data_feeds.data_feed_orchestrator.get_multiple_quotes()](data_feeds/data_feed_orchestrator.py:1534)
- get_financial_statements(symbol)
  - Ref: [data_feeds.data_feed_orchestrator.get_financial_statements()](data_feeds/data_feed_orchestrator.py:1538)
- Market breadth and FinViz unified access:
  - get_market_breadth(), get_sector_performance(), get_finviz_news(), get_finviz_insider_trading(), get_finviz_earnings(), get_finviz_forex(), get_finviz_crypto()
  - Refs: [data_feeds.data_feed_orchestrator.get_market_breadth()](data_feeds/data_feed_orchestrator.py:1576), [data_feeds.data_feed_orchestrator.get_sector_performance()](data_feeds/data_feed_orchestrator.py:1580), [data_feeds.data_feed_orchestrator.get_finviz_news()](data_feeds/data_feed_orchestrator.py:1584), [data_feeds.data_feed_orchestrator.get_finviz_insider_trading()](data_feeds/data_feed_orchestrator.py:1588), [data_feeds.data_feed_orchestrator.get_finviz_earnings()](data_feeds/data_feed_orchestrator.py:1592), [data_feeds.data_feed_orchestrator.get_finviz_forex()](data_feeds/data_feed_orchestrator.py:1596), [data_feeds.data_feed_orchestrator.get_finviz_crypto()](data_feeds/data_feed_orchestrator.py:1600)

## 3) ConsolidatedDataFeed Endpoints (Orchestrator Delegation Targets)

Class: [data_feeds.consolidated_data_feed.ConsolidatedDataFeed](data_feeds/consolidated_data_feed.py:909)

Core methods:
- get_quote(symbol): yfinance → fmp → finnhub → investiny → stockdex
  - Ref: [data_feeds.consolidated_data_feed.ConsolidatedDataFeed.get_quote()](data_feeds/consolidated_data_feed.py:932)
- get_historical(symbol, period="1y", from_date=None, to_date=None)
  - yfinance or fmp/investiny/stockdex
  - Ref: [data_feeds.consolidated_data_feed.ConsolidatedDataFeed.get_historical()](data_feeds/consolidated_data_feed.py:947)
- get_company_info(symbol): yfinance → fmp → finnhub → investiny → stockdex
  - Ref: [data_feeds.consolidated_data_feed.ConsolidatedDataFeed.get_company_info()](data_feeds/consolidated_data_feed.py:968)
- get_news(symbol, limit=10): finnhub → yfinance
  - Ref: [data_feeds.consolidated_data_feed.ConsolidatedDataFeed.get_news()](data_feeds/consolidated_data_feed.py:983)
- get_multiple_quotes(symbols)
  - Ref: [data_feeds.consolidated_data_feed.ConsolidatedDataFeed.get_multiple_quotes()](data_feeds/consolidated_data_feed.py:999)
- get_financial_statements(symbol): Stockdex first, else {}
  - Ref: [data_feeds.consolidated_data_feed.ConsolidatedDataFeed.get_financial_statements()](data_feeds/consolidated_data_feed.py:1028)

Supporting adapters include specific endpoints:
- YFinanceAdapter.get_news(symbol, limit) – Yahoo News via yfinance
  - Ref: [data_feeds.consolidated_data_feed.YFinanceAdapter.get_news()](data_feeds/consolidated_data_feed.py:252)
- FinnhubAdapter.get_news(symbol, limit)
  - Ref: [data_feeds.consolidated_data_feed.FinnhubAdapter.get_news()](data_feeds/consolidated_data_feed.py:319)

## 4) Unified Facade (data_feeds_unified)

Class: [data_feeds.data_feeds_unified.UnifiedDataProvider](data_feeds/data_feeds_unified.py:42)

Facade endpoints (legacy-compatible shapes), all are orchestrator-backed:
- get_stock_price(symbol) – wraps orchestrator get_quote
  - Ref: [data_feeds.data_feeds_unified.UnifiedDataProvider.get_stock_price()](data_feeds/data_feeds_unified.py:53)
- get_stock_data(symbol, period="1y") – wraps orchestrator get_market_data, returns DataFrame
  - Ref: [data_feeds.data_feeds_unified.UnifiedDataProvider.get_stock_data()](data_feeds/data_feeds_unified.py:59)
- get_real_time_quote(symbol) – dict form
  - Ref: [data_feeds.data_feeds_unified.UnifiedDataProvider.get_real_time_quote()](data_feeds/data_feeds_unified.py:68)
- get_multiple_quotes(symbols) – dict per symbol
  - Ref: [data_feeds.data_feeds_unified.UnifiedDataProvider.get_multiple_quotes()](data_feeds/data_feeds_unified.py:83)
- get_company_profile(symbol)
  - Ref: [data_feeds.data_feeds_unified.UnifiedDataProvider.get_company_profile()](data_feeds/data_feeds_unified.py:100)
- get_company_news(symbol, limit=10)
  - Ref: [data_feeds.data_feeds_unified.UnifiedDataProvider.get_company_news()](data_feeds/data_feeds_unified.py:120)
- get_ohlcv_data(symbol, period="1y") – DF
  - Ref: [data_feeds.data_feeds_unified.UnifiedDataProvider.get_ohlcv_data()](data_feeds/data_feeds_unified.py:135)
- calculate_technical_indicators(symbol, period="6mo")
  - Ref: [data_feeds.data_feeds_unified.UnifiedDataProvider.calculate_technical_indicators()](data_feeds/data_feeds_unified.py:140)
- Portfolio helpers: get_portfolio_data, monitor_portfolio_risk, cache ops
  - Refs: [data_feeds.data_feeds_unified.UnifiedDataProvider.get_portfolio_data()](data_feeds/data_feeds_unified.py:198), [data_feeds.data_feeds_unified.UnifiedDataProvider.monitor_portfolio_risk()](data_feeds/data_feeds_unified.py:243), [data_feeds.data_feeds_unified.UnifiedDataProvider.get_cache_stats()](data_feeds/data_feeds_unified.py:268), [data_feeds.data_feeds_unified.UnifiedDataProvider.clear_cache()](data_feeds/data_feeds_unified.py:273)

## 5) Oracle Data Interface (oracle_data_interface)

Class: [data_feeds.oracle_data_interface.OracleDataProvider](data_feeds/oracle_data_interface.py:39)

High-level endpoints (compose orchestrator endpoints):
- get_comprehensive_market_intelligence(tickers=None)
  - Aggregates quotes, sentiment, and technicals
  - Ref: [data_feeds.oracle_data_interface.OracleDataProvider.get_comprehensive_market_intelligence()](data_feeds/oracle_data_interface.py:50)
- get_market_internals()
  - Uses orchestrator quotes; provides summary incl. VIX
  - Ref: [data_feeds.oracle_data_interface.OracleDataProvider.get_market_internals()](data_feeds/oracle_data_interface.py:116)
- get_earnings_calendar(tickers=None) – basic structure, TODO real data
  - Ref: [data_feeds.oracle_data_interface.OracleDataProvider.get_earnings_calendar()](data_feeds/oracle_data_interface.py:146)
- get_options_analysis(tickers=None) – basic heuristic using quote volumes
  - Ref: [data_feeds.oracle_data_interface.OracleDataProvider.get_options_analysis()](data_feeds/oracle_data_interface.py:168)
- get_sentiment_analysis(tickers=None)
  - Aggregates orchestrator get_sentiment_data outputs
  - Ref: [data_feeds.oracle_data_interface.OracleDataProvider.get_sentiment_analysis()](data_feeds/oracle_data_interface.py:199)
- validate_data_quality()
  - Ref: [data_feeds.oracle_data_interface.OracleDataProvider.validate_data_quality()](data_feeds/oracle_data_interface.py:253)

Module-level compatibility functions:
- fetch_market_internals(), fetch_options_flow(), fetch_dark_pool_data(), fetch_sentiment_data(), fetch_earnings_calendar(), get_signals_from_scrapers_v2()
  - Refs: [data_feeds.oracle_data_interface.fetch_market_internals()](data_feeds/oracle_data_interface.py:307), [data_feeds.oracle_data_interface.fetch_options_flow()](data_feeds/oracle_data_interface.py:312), [data_feeds.oracle_data_interface.fetch_dark_pool_data()](data_feeds/oracle_data_interface.py:316), [data_feeds.oracle_data_interface.fetch_sentiment_data()](data_feeds/oracle_data_interface.py:328), [data_feeds.oracle_data_interface.fetch_earnings_calendar()](data_feeds/oracle_data_interface.py:332), [data_feeds.oracle_data_interface.get_signals_from_scrapers_v2()](data_feeds/oracle_data_interface.py:336)

## 6) Duplicate Endpoint Coverage Notes

News endpoints (duplicate category):
- Yahoo Finance News
  - Path(s):
    - Consolidated: [data_feeds.consolidated_data_feed.YFinanceAdapter.get_news()](data_feeds/consolidated_data_feed.py:252)
    - Orchestrator Advanced Sentiment aggregation via [data_feeds.news_scraper.fetch_headlines_yahoo_finance()](data_feeds/data_feed_orchestrator.py:1201)
  - Category: Company news/headlines
  - Notes: Yahoo headlines inform advanced sentiment aggregator and can overlap with FinViz headlines semantically

- Finnhub Company News
  - Path: [data_feeds.consolidated_data_feed.FinnhubAdapter.get_news()](data_feeds/consolidated_data_feed.py:319)
  - Category: Company news; consolidation path prefers Finnhub first for get_news()

- FinViz News
  - Paths:
    - Orchestrator FinViz news collection: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_news()](data_feeds/data_feed_orchestrator.py:1399)
    - Advanced sentiment aggregation reads FinViz headlines via FinViz adapter: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_advanced_sentiment_data()](data_feeds/data_feed_orchestrator.py:1208)
  - Category: Market-wide and ticker-related headlines
  - Duplicate note: FinViz headlines are conceptually similar to Yahoo headlines. Cataloged duplicates: “Yahoo Finance headlines” vs “FinViz headlines”. Downstream consumers should deduplicate by title/url/timestamp when merging feeds.

Sentiment endpoints (potential overlap):
- Reddit Sentiment
  - Orchestrator direct: [data_feeds.data_feed_orchestrator.RedditAdapter.get_sentiment()](data_feeds/data_feed_orchestrator.py:537)
  - Advanced sentiment aggregates Reddit sample_texts again: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_advanced_sentiment_data()](data_feeds/data_feed_orchestrator.py:1164)
  - Duplicate note: Advanced sentiment is an aggregate; treat it as derived endpoint. When combining with raw Reddit sentiment, avoid double-counting signals.

- Twitter Sentiment
  - Orchestrator direct: [data_feeds.data_feed_orchestrator.TwitterAdapter.get_sentiment()](data_feeds/data_feed_orchestrator.py:664)
  - Advanced sentiment aggregates tweets again: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_advanced_sentiment_data()](data_feeds/data_feed_orchestrator.py:1181)
  - Duplicate note: Same deduplication guidance as Reddit.

Quotes and Historical data (overlapping providers):
- Quotes via yfinance, FMP, Finnhub, Investiny, Stockdex
  - Consolidated: fallback chain in [data_feeds.consolidated_data_feed.ConsolidatedDataFeed.get_quote()](data_feeds/consolidated_data_feed.py:932)
  - Orchestrator: primary yfinance, optional wrappers (TwelveData can be preferred)
  - Duplicate note: Orchestrator selects best-quality; consolidated fallback ensures one final output. If both layers are queried independently, may yield overlapping but not identical values. Prefer orchestrator’s unified access.

- Historical via yfinance, FMP, Investiny, Stockdex
  - Consolidated: [data_feeds.consolidated_data_feed.ConsolidatedDataFeed.get_historical()](data_feeds/consolidated_data_feed.py:947)
  - Orchestrator: [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_market_data()](data_feeds/data_feed_orchestrator.py:967)
  - Duplicate note: Choose a single path for consistency (recommend orchestrator), since validation/caching differ.

FinViz thematic datasets:
- Datasets: market breadth, sector performance, news, insider trading, earnings, forex, crypto
  - Orchestrator endpoints:
    - [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_market_breadth()](data_feeds/data_feed_orchestrator.py:1341)
    - [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_sector_performance()](data_feeds/data_feed_orchestrator.py:1370)
    - [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_news()](data_feeds/data_feed_orchestrator.py:1399)
    - [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_insider_trading()](data_feeds/data_feed_orchestrator.py:1423)
    - [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_earnings()](data_feeds/data_feed_orchestrator.py:1448)
    - [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_forex()](data_feeds/data_feed_orchestrator.py:1473)
    - [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_finviz_crypto()](data_feeds/data_feed_orchestrator.py:1498)
  - Duplicate note: FinViz news overlaps with Yahoo/Finnhub news categories; other FinViz datasets are unique.

## 7) Endpoint Index (Quick Reference)

Orchestrator module-level (preferred external surface):
- Quotes: [data_feeds.data_feed_orchestrator.get_quote()](data_feeds/data_feed_orchestrator.py:1556)
- Market data: [data_feeds.data_feed_orchestrator.get_market_data()](data_feeds/data_feed_orchestrator.py:1560)
- Sentiment (sources): [data_feeds.data_feed_orchestrator.get_sentiment_data()](data_feeds/data_feed_orchestrator.py:1564)
- Advanced sentiment: [data_feeds.data_feed_orchestrator.get_advanced_sentiment()](data_feeds/data_feed_orchestrator.py:1568)
- Company info: [data_feeds.data_feed_orchestrator.get_company_info()](data_feeds/data_feed_orchestrator.py:1526)
- Company news: [data_feeds.data_feed_orchestrator.get_news()](data_feeds/data_feed_orchestrator.py:1530)
- Multiple quotes: [data_feeds.data_feed_orchestrator.get_multiple_quotes()](data_feeds/data_feed_orchestrator.py:1534)
- Financial statements: [data_feeds.data_feed_orchestrator.get_financial_statements()](data_feeds/data_feed_orchestrator.py:1538)
- System health: [data_feeds.data_feed_orchestrator.get_system_health()](data_feeds/data_feed_orchestrator.py:1572)
- FinViz datasets:
  - Breadth: [data_feeds.data_feed_orchestrator.get_market_breadth()](data_feeds/data_feed_orchestrator.py:1576)
  - Sector performance: [data_feeds.data_feed_orchestrator.get_sector_performance()](data_feeds/data_feed_orchestrator.py:1580)
  - News: [data_feeds.data_feed_orchestrator.get_finviz_news()](data_feeds/data_feed_orchestrator.py:1584)
  - Insider: [data_feeds.data_feed_orchestrator.get_finviz_insider_trading()](data_feeds/data_feed_orchestrator.py:1588)
  - Earnings: [data_feeds.data_feed_orchestrator.get_finviz_earnings()](data_feeds/data_feed_orchestrator.py:1592)
  - Forex: [data_feeds.data_feed_orchestrator.get_finviz_forex()](data_feeds/data_feed_orchestrator.py:1596)
  - Crypto: [data_feeds.data_feed_orchestrator.get_finviz_crypto()](data_feeds/data_feed_orchestrator.py:1600)

## 8) Deduplication Guidance

- When aggregating “news”, de-duplicate across:
  - Yahoo Finance (yfinance ticker.news, headlines scraper) and FinViz headlines
  - Finnhub company_news when combined with Yahoo or FinViz
  - Suggested keys: normalized title, url host+path, and published timestamp (rounded to minute)

- When combining “sentiment”:
  - Keep Advanced Sentiment as a derived aggregate. Avoid summing it with raw Reddit/Twitter scores for the same texts to prevent double-counting.
  - If raw and advanced are both needed, maintain separation and annotate provenance.

- Prefer orchestrator module-level functions for external consumers to benefit from caching, rate limiting, quality scoring, and performance tracking.

---

# Provider Wrapper and Adapter Endpoint Matrix

This section expands the catalog by analyzing each wrapper/adapter and listing all endpoints provided by each API or scraper, with explicit capability coverage and references.

## A) Standardized Wrapper Adapters (SourceAdapterProtocol)

File: [data_feeds.adapter_wrappers.py](data_feeds/adapter_wrappers.py:1)

Common protocol methods: capabilities(), fetch_quote(), fetch_historical(), fetch_company_info(), fetch_news(), fetch_sentiment(), health().

1. YFinanceAdapterWrapper
   - Class: [data_feeds.adapter_wrappers.YFinanceAdapterWrapper](data_feeds/adapter_wrappers.py:98)
   - Capabilities: {"quote","historical","company_info","news"} via [adapter_wrappers.YFinanceAdapterWrapper.capabilities()](data_feeds/adapter_wrappers.py:104)
   - Endpoints:
     - fetch_quote(symbol) → Quote
       - [adapter_wrappers.YFinanceAdapterWrapper.fetch_quote()](data_feeds/adapter_wrappers.py:107)
     - fetch_historical(symbol, period, interval?, from_date?, to_date?) → DataFrame
       - Delegates to consolidated YF get_historical(symbol, period)
       - [adapter_wrappers.YFinanceAdapterWrapper.fetch_historical()](data_feeds/adapter_wrappers.py:115)
     - fetch_company_info(symbol) → CompanyInfo
       - [adapter_wrappers.YFinanceAdapterWrapper.fetch_company_info()](data_feeds/adapter_wrappers.py:131)
     - fetch_news(symbol, limit) → List[NewsItem]
       - [adapter_wrappers.YFinanceAdapterWrapper.fetch_news()](data_feeds/adapter_wrappers.py:139)
     - fetch_sentiment(symbol, …)
       - Not supported; raises NotImplementedError
       - [adapter_wrappers.YFinanceAdapterWrapper.fetch_sentiment()](data_feeds/adapter_wrappers.py:147)
     - health() → minimal rate/usage metadata
       - [adapter_wrappers._BaseWrapper.health()](data_feeds/adapter_wrappers.py:77)

2. FMPAdapterWrapper
   - Class: [data_feeds.adapter_wrappers.FMPAdapterWrapper](data_feeds/adapter_wrappers.py:151)
   - Capabilities: {"quote","historical","company_info"} via [adapter_wrappers.FMPAdapterWrapper.capabilities()](data_feeds/adapter_wrappers.py:156)
   - Endpoints:
     - fetch_quote(symbol) → Quote
       - [adapter_wrappers.FMPAdapterWrapper.fetch_quote()](data_feeds/adapter_wrappers.py:159)
     - fetch_historical(symbol, from_date, to_date) → DataFrame
       - [adapter_wrappers.FMPAdapterWrapper.fetch_historical()](data_feeds/adapter_wrappers.py:167)
     - fetch_company_info(symbol) → CompanyInfo
       - [adapter_wrappers.FMPAdapterWrapper.fetch_company_info()](data_feeds/adapter_wrappers.py:183)
     - fetch_news(...) – Not supported (NotImplementedError)
       - [adapter_wrappers.FMPAdapterWrapper.fetch_news()](data_feeds/adapter_wrappers.py:191)
     - fetch_sentiment(...) – Not supported (NotImplementedError)
       - [adapter_wrappers.FMPAdapterWrapper.fetch_sentiment()](data_feeds/adapter_wrappers.py:194)

3. FinnhubAdapterWrapper
   - Class: [data_feeds.adapter_wrappers.FinnhubAdapterWrapper](data_feeds/adapter_wrappers.py:198)
   - Capabilities: {"quote","company_info","news"} via [adapter_wrappers.FinnhubAdapterWrapper.capabilities()](data_feeds/adapter_wrappers.py:203)
   - Endpoints:
     - fetch_quote(symbol) → Quote
       - [adapter_wrappers.FinnhubAdapterWrapper.fetch_quote()](data_feeds/adapter_wrappers.py:206)
     - fetch_historical(...) – Not supported here (NotImplementedError)
       - [adapter_wrappers.FinnhubAdapterWrapper.fetch_historical()](data_feeds/adapter_wrappers.py:214)
     - fetch_company_info(symbol) → CompanyInfo
       - [adapter_wrappers.FinnhubAdapterWrapper.fetch_company_info()](data_feeds/adapter_wrappers.py:224)
     - fetch_news(symbol, limit) → List[NewsItem]
       - [adapter_wrappers.FinnhubAdapterWrapper.fetch_news()](data_feeds/adapter_wrappers.py:232)
     - fetch_sentiment(...) – Not supported (NotImplementedError)
       - [adapter_wrappers.FinnhubAdapterWrapper.fetch_sentiment()](data_feeds/adapter_wrappers.py:240)

4. FinanceDatabaseAdapterWrapper
   - Class: [data_feeds.adapter_wrappers.FinanceDatabaseAdapterWrapper](data_feeds/adapter_wrappers.py:244)
   - Capabilities: {"fundamentals"} (primarily search/discovery) via [adapter_wrappers.FinanceDatabaseAdapterWrapper.capabilities()](data_feeds/adapter_wrappers.py:249)
   - Endpoints:
     - fetch_quote / fetch_historical / fetch_company_info / fetch_news / fetch_sentiment – Not supported here (NotImplementedError)
       - Refs: [data_feeds/adapter_wrappers.py lines 253-275](data_feeds/adapter_wrappers.py:253)
     - search_equities(**kwargs) → Dict
       - [adapter_wrappers.FinanceDatabaseAdapterWrapper.search_equities()](data_feeds/adapter_wrappers.py:277)
     - search_etfs(**kwargs) → Dict
       - [adapter_wrappers.FinanceDatabaseAdapterWrapper.search_etfs()](data_feeds/adapter_wrappers.py:285)
   - Note: Provides discovery/fundamentals search rather than per-symbol OHLC or quotes.

Provider-level duplicate notes:
- News: Both YFinance (Yahoo) and Finnhub wrappers expose news; duplicates must be deduped if both are queried.
- Quotes/Company Info: YFinance and Finnhub/FMP overlap; orchestrator should select by quality/availability.

## B) TwelveData Adapter

File: [data_feeds.twelvedata_adapter.TwelveDataAdapter](data_feeds/twelvedata_adapter.py:64)

Capabilities exposed in this project:
- get_quote(symbol) → Quote-like object (orchestrator Quote model)
  - [twelvedata_adapter.TwelveDataAdapter.get_quote()](data_feeds/twelvedata_adapter.py:100)
- get_market_data(symbol, period="1y", interval="1d", outputsize=None, start=None, end=None) → MarketData
  - Interval/period alias normalization and mapping
  - [twelvedata_adapter.TwelveDataAdapter.get_market_data()](data_feeds/twelvedata_adapter.py:148)

Provider duplicate notes:
- Quotes and Historical overlap with YFinance and consolidated providers. Orchestrator can prefer Twelve Data if explicitly requested in preferred_sources.
- Twelve Data does not provide news/sentiment in this adapter.

## C) FinViz Adapter

File: [data_feeds.finviz_adapter.FinVizAdapter](data_feeds/finviz_adapter.py:18)

Capabilities (market/instrument aggregates; scraping via finviz_scraper):
- get_market_breadth() → MarketBreadth
  - [finviz_adapter.FinVizAdapter.get_market_breadth()](data_feeds/finviz_adapter.py:22)
- get_sector_performance() → List[GroupPerformance]
  - [finviz_adapter.FinVizAdapter.get_sector_performance()](data_feeds/finviz_adapter.py:41)
- get_news() → Dict[str, pd.DataFrame] with keys: 'news', 'blogs'
  - [finviz_adapter.FinVizAdapter.get_news()](data_feeds/finviz_adapter.py:60)
- get_insider_trading() → pd.DataFrame
  - [finviz_adapter.FinVizAdapter.get_insider_trading()](data_feeds/finviz_adapter.py:67)
- get_earnings() → Dict[str, pd.DataFrame]
  - [finviz_adapter.FinVizAdapter.get_earnings()](data_feeds/finviz_adapter.py:74)
- get_forex() → pd.DataFrame
  - [finviz_adapter.FinVizAdapter.get_forex()](data_feeds/finviz_adapter.py:81)
- get_crypto() → pd.DataFrame
  - [finviz_adapter.FinVizAdapter.get_crypto()](data_feeds/finviz_adapter.py:88)

Duplicate notes:
- FinViz news vs Yahoo (YFinance/Headline scraper) vs Finnhub – content overlap likely.
- Other FinViz datasets (breadth, sector performance, insider, earnings calendar, forex, crypto) are unique categories within this codebase.

## D) Twitter Sentiment Feed

File: [data_feeds.twitter_feed.TwitterSentimentFeed](data_feeds/twitter_feed.py:39)

Capabilities:
- fetch(query, limit=100) → List[Dict] tweets with VADER (and optional TextBlob) sentiment, plus extracted tickers/lang
  - [twitter_feed.TwitterSentimentFeed.fetch()](data_feeds/twitter_feed.py:154)
- Provider is used by orchestrator TwitterAdapter.get_sentiment(symbol, limit) to build a SentimentData aggregate
  - Orchestrator path: [data_feeds.data_feed_orchestrator.TwitterAdapter.get_sentiment()](data_feeds/data_feed_orchestrator.py:664)

Duplicate notes:
- Twitter sentiment contributes to both raw per-source sentiment and Advanced Sentiment aggregate. Avoid double-counting.

## E) Yahoo Finance Headlines Scraper

File: [data_feeds.news_scraper.fetch_headlines_yahoo_finance](data_feeds/news_scraper.py:7)

Capabilities:
- fetch_headlines_yahoo_finance() → List[str] headlines from https://finance.yahoo.com/
  - [news_scraper.fetch_headlines_yahoo_finance()](data_feeds/news_scraper.py:7)
- Used inside orchestrator advanced sentiment aggregator to broaden news text corpus
  - [data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_advanced_sentiment_data()](data_feeds/data_feed_orchestrator.py:1197)

Duplicate notes:
- Overlaps category-wise with FinViz news and Finnhub company news. Use deduplication across headline/title/url/timestamp.

---

# Provider-by-Provider Endpoint Summary

This quick matrix shows for each provider/wrapper which operations are available in-code:

- YFinance (Wrapper):
  - Quote: Yes
  - Historical: Yes
  - Company Info: Yes
  - News: Yes (Yahoo company news)
  - Sentiment: No
- FMP (Wrapper):
  - Quote: Yes
  - Historical: Yes (date-range)
  - Company Info: Yes
  - News: No
  - Sentiment: No
- Finnhub (Wrapper):
  - Quote: Yes
  - Historical: No (not exposed here)
  - Company Info: Yes
  - News: Yes (company news)
  - Sentiment: No
- FinanceDatabase (Wrapper):
  - Discovery/Fundamentals Search: Yes (search_equities / search_etfs)
  - Quote/Historical/Company Info/News/Sentiment: Not via this wrapper
- TwelveData (Direct Adapter):
  - Quote: Yes
  - Historical (time_series): Yes (interval mapping, outputsize calc)
  - Company Info: Not in this adapter
  - News: No
  - Sentiment: No
- FinViz (Direct Adapter):
  - Breadth: Yes
  - Sector Performance: Yes
  - News/Blogs: Yes
  - Insider Trading: Yes
  - Earnings: Yes
  - Forex: Yes
  - Crypto: Yes
- Reddit (via orchestrator adapter):
  - Sentiment: Yes (batch fetch via fetch_reddit_sentiment)
- Twitter (via TwitterSentimentFeed + orchestrator adapter):
  - Sentiment: Yes (VADER/TextBlob-based per tweet, aggregated)

---

# Notes on Integration Strategy

- Prefer orchestrator module-level endpoints for external consumption to leverage caching (SmartCache), rate limiting (RateLimiter), performance tracking (PerformanceTracker), and quality validation (DataValidator).
- To avoid duplicates:
  - For news: select a single provider or dedupe merged sets using title/url/timestamp keys.
  - For sentiment: treat Advanced Sentiment as derived; keep raw Reddit/Twitter separate when needed for diagnostics.
- For historical data with intraday intervals, Twelve Data provides richer interval support in this codebase; yfinance remains default daily/1h source through wrappers/adapters.
