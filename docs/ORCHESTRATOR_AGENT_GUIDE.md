# Oracle-X DataFeedOrchestrator - Complete Agent Reference Guide

## 🚀 Quick Start

The DataFeedOrchestrator is Oracle-X's unified financial data interface, providing real-time quotes, market data, sentiment analysis, and financial calculations with intelligent fallback and quality validation.

```python
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator

# Initialize the orchestrator
orchestrator = DataFeedOrchestrator()

# Get a stock quote
quote = orchestrator.get_quote("AAPL")
print(f"AAPL Price: ${quote.price}")

# Get sentiment data
sentiment = orchestrator.get_sentiment_data("AAPL")
print(f"Sentiment Score: {sentiment['reddit'].sentiment_score}")
```

## 📊 Core Data Functions

### 1. Real-Time Stock Quotes

#### `get_quote(symbol: str, preferred_sources: Optional[List[DataSource]] = None) -> Optional[Quote]`

**Purpose**: Get real-time stock quote with intelligent source fallback
**Performance**: ~384ms average response time
**Quality**: Automatic validation with 80-100 quality scores

**Returns**: Quote object with:
- `symbol`: Stock symbol
- `price`: Current price (Decimal)
- `change`: Price change (Decimal)
- `change_percent`: Percentage change (Decimal)
- `volume`: Trading volume (int)
- `market_cap`: Market capitalization (int)
- `pe_ratio`: Price-to-earnings ratio (Decimal)
- `day_low`/`day_high`: Daily range (Decimal)
- `year_low`/`year_high`: 52-week range (Decimal)
- `timestamp`: Data timestamp (datetime)
- `source`: Data source used (str)
- `quality_score`: Quality validation score 0-100 (float)

**Agent Example**:
```python
# Basic quote request
quote = orchestrator.get_quote("AAPL")
if quote:
    print(f"✅ {quote.symbol}: ${quote.price} ({quote.change_percent:+.2f}%)")
    print(f"📊 Volume: {quote.volume:,} | Source: {quote.source}")
    print(f"🎯 Quality Score: {quote.quality_score}/100")

# With preferred sources
quote = orchestrator.get_quote("TSLA", preferred_sources=[DataSource.YFINANCE, DataSource.TWELVE_DATA])
```

**Data Sources**: YFinance (primary), TwelveData (fallback), FMP, Finnhub
**Fallback Logic**: Automatic failover with performance tracking

---

### 2. Historical Market Data

#### `get_market_data(symbol: str, period: str = "1y", interval: str = "1d", preferred_sources: Optional[List[DataSource]] = None) -> Optional[MarketData]`

**Purpose**: Get historical OHLCV data with quality validation
**Performance**: ~595ms average response time
**Quality**: 100/100 quality score for YFinance data

**Parameters**:
- `period`: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
- `interval`: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"

**Returns**: MarketData object with:
- `symbol`: Stock symbol
- `data`: pandas DataFrame with OHLCV data
- `timeframe`: Period and interval (e.g., "1mo_1d")
- `source`: Data source used
- `timestamp`: Retrieval timestamp
- `quality_score`: Quality validation score

**Agent Example**:
```python
# Get 1 month of daily data
market_data = orchestrator.get_market_data("AAPL", period="1mo", interval="1d")
if market_data:
    df = market_data.data
    print(f"📈 {market_data.symbol} - {len(df)} days of data")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"💯 Quality Score: {market_data.quality_score}/100")
    print(f"📅 Latest Close: ${df['Close'].iloc[-1]:.2f}")

# Get intraday 5-minute data
intraday = orchestrator.get_market_data("SPY", period="1d", interval="5m")
```

**Data Sources**: YFinance (primary), TwelveData (fallback), Investiny
**Quality Validation**: Missing data detection, outlier analysis, completeness scoring

---

### 3. Multi-Source Sentiment Analysis

#### `get_sentiment_data(symbol: str, sources: Optional[List[DataSource]] = None) -> Dict[str, SentimentData]`

**Purpose**: Aggregate sentiment from multiple sources with confidence scoring
**Performance**: ~4.1s total (Reddit: 1.86s, Twitter: 1.84s, Yahoo: 0.42s)
**Quality**: FinBERT-enhanced sentiment with confidence metrics

**Returns**: Dictionary of SentimentData objects with:
- `symbol`: Stock symbol
- `sentiment_score`: Sentiment score -1 to +1 (float)
- `confidence`: Confidence level 0-1 (float)
- `source`: Data source name (str)
- `timestamp`: Analysis timestamp (datetime)
- `sample_size`: Number of samples analyzed (int)
- `raw_data`: Additional source-specific data (dict)

**Agent Example**:
```python
# Get sentiment from all sources
sentiment_data = orchestrator.get_sentiment_data("AAPL")
for source, sentiment in sentiment_data.items():
    score = sentiment.sentiment_score
    confidence = sentiment.confidence
    samples = sentiment.sample_size or 0

    trend = "🟢 Bullish" if score > 0.1 else "🔴 Bearish" if score < -0.1 else "🟡 Neutral"
    print(f"{source}: {score:+.3f} ({confidence:.3f} confidence, {samples} samples) {trend}")

# Reddit sentiment: +0.992 (0.520 confidence, 8 tickers) 🟢 Bullish
# Twitter sentiment: +0.330 (0.439 confidence, 20 samples) 🟢 Bullish
# Yahoo sentiment: +0.150 (0.850 confidence, 15 samples) 🟢 Bullish
```

**Data Sources**: Reddit (8+ tickers), Twitter (FinBERT-enhanced), Yahoo News
**Caching**: 5-minute TTL on Reddit data for 30% performance improvement

---

### 4. Company Information

#### `get_company_info(symbol: str) -> Optional[CompanyInfo]`

**Purpose**: Get comprehensive company fundamentals and information

**Returns**: CompanyInfo object with:
- Basic company details (name, description, industry, sector)
- Financial metrics (market cap, PE ratio, dividend yield)
- Company officers and key statistics
- Business summary and company profile

**Agent Example**:
```python
info = orchestrator.get_company_info("AAPL")
if info:
    print(f"🏢 {info.name} ({info.symbol})")
    print(f"🏭 Industry: {info.industry} | Sector: {info.sector}")
    print(f"💰 Market Cap: ${info.market_cap/1e9:.1f}B")
    print(f"📈 P/E Ratio: {info.pe_ratio}")
```

---

### 5. Latest News

#### `get_news(symbol: str, limit: int = 10) -> List[NewsItem]`

**Purpose**: Get latest news articles for a symbol

**Returns**: List of NewsItem objects with:
- `title`: Article headline
- `summary`: Article summary
- `url`: Article URL
- `timestamp`: Publication time
- `source`: News source

**Agent Example**:
```python
news = orchestrator.get_news("AAPL", limit=5)
for article in news[:3]:
    print(f"📰 {article.title}")
    print(f"🕒 {article.timestamp} | 🌐 {article.source}")
    print(f"📝 {article.summary}\n")
```

---

## 📈 Advanced Analytics Functions

### 6. Market Breadth Analysis

#### `get_market_breadth() -> Optional[MarketBreadth]`

**Purpose**: Get market-wide statistics and breadth indicators
**Performance**: ~305ms response time

**Returns**: MarketBreadth object with:
- `advancers`: Number of advancing stocks (int)
- `decliners`: Number of declining stocks (int)
- `new_highs`: New 52-week highs (int)
- `new_lows`: New 52-week lows (int)
- `advance_decline_ratio`: A/D ratio (float)

**Agent Example**:
```python
breadth = orchestrator.get_market_breadth()
if breadth:
    print(f"📊 Market Breadth Analysis")
    print(f"⬆️  Advancers: {breadth.advancers:,}")
    print(f"⬇️  Decliners: {breadth.decliners:,}")
    print(f"🆕 New Highs: {breadth.new_highs}")
    print(f"🔻 New Lows: {breadth.new_lows}")
    print(f"📈 A/D Ratio: {breadth.advance_decline_ratio:.2f}")
```

**Current Sample**: 2,110 advancers, 3,213 decliners, 108 new highs, 104 new lows

---

### 7. Sector Performance

#### `get_sector_performance() -> List[GroupPerformance]`

**Purpose**: Get performance metrics for all market sectors
**Performance**: ~136ms response time

**Returns**: List of GroupPerformance objects with:
- `name`: Sector name (str)
- `performance_1d`: 1-day performance (float)
- `performance_1w`: 1-week performance (float)
- `performance_1m`: 1-month performance (float)
- `performance_ytd`: Year-to-date performance (float)

**Agent Example**:
```python
sectors = orchestrator.get_sector_performance()
print("🏭 Sector Performance (1-Day)")
for sector in sorted(sectors, key=lambda x: x.performance_1d, reverse=True):
    perf = sector.performance_1d * 100
    emoji = "🟢" if perf > 0 else "🔴" if perf < 0 else "🟡"
    print(f"{emoji} {sector.name}: {perf:+.2f}%")
```

**Current Sample**: 11 sectors tracked including Basic Materials, Technology, Healthcare

---

### 8. Earnings Calendar

#### `get_earnings_calendar_detailed(tickers: Optional[List[str]] = None) -> Optional[List[dict]]`

**Purpose**: Get detailed earnings calendar with estimates and actuals

**Agent Example**:
```python
# All upcoming earnings
earnings = orchestrator.get_earnings_calendar_detailed()

# Specific tickers
earnings = orchestrator.get_earnings_calendar_detailed(["AAPL", "MSFT", "GOOGL"])
for event in earnings[:5]:
    print(f"📅 {event['symbol']} - {event['date']}")
    print(f"💰 EPS Estimate: ${event['eps_estimate']}")
```

---

### 9. Options Analytics

#### `get_options_analytics(symbol: str, include: Optional[List[str]] = None) -> Optional[dict]`

**Purpose**: Get options data and analytics for a symbol

**Agent Example**:
```python
options = orchestrator.get_options_analytics("AAPL", include=["chains", "volumes", "greeks"])
if options:
    print(f"📈 Options Analytics for {options['symbol']}")
    print(f"🔀 Put/Call Ratio: {options.get('put_call_ratio', 'N/A')}")
```

---

### 10. Financial Statements

#### `get_financial_statements(symbol: str) -> Dict[str, Optional[pd.DataFrame]]`

**Purpose**: Get comprehensive financial statements

**Returns**: Dictionary with:
- `balance_sheet`: Balance sheet data (DataFrame)
- `income_statement`: Income statement data (DataFrame)
- `cash_flow`: Cash flow statement data (DataFrame)

**Agent Example**:
```python
statements = orchestrator.get_financial_statements("AAPL")
for statement_type, data in statements.items():
    if data is not None and not data.empty:
        print(f"📊 {statement_type}: {len(data)} periods")
        print(f"📋 Metrics: {list(data.index)[:5]}...")
```

---

## 🧮 Financial Calculator Functions

### Technical Indicators and Calculations

The Oracle-X system includes a comprehensive financial calculator with technical indicators:

```python
from data_feeds.financial_calculator import FinancialCalculator

# Get quote and market data
quote = orchestrator.get_quote("AAPL")
market_data = orchestrator.get_market_data("AAPL", period="1mo")

# Calculate comprehensive financial metrics
metrics = FinancialCalculator.calculate_comprehensive_metrics(quote, market_data.data)

print(f"💰 Price: {FinancialCalculator.format_currency(metrics.price)}")
print(f"📊 RSI (14): {metrics.rsi_14:.2f}")
print(f"📈 SMA (20): {FinancialCalculator.format_currency(metrics.sma_20)}")
print(f"📉 SMA (50): {FinancialCalculator.format_currency(metrics.sma_50)}")
print(f"📊 Volatility (1D): {metrics.volatility_1d:.4f}")
print(f"💹 Market Cap: {FinancialCalculator.calculate_market_cap_billions(metrics.market_cap):.1f}B")
```

**Available Calculations**:
- **Technical Indicators**: RSI, SMA (20/50), Bollinger Bands
- **Volatility Metrics**: 1-day and 30-day volatility
- **Portfolio Analytics**: Correlation analysis, portfolio metrics
- **Formatting**: Currency formatting, percentage calculations
- **Quality Scoring**: Data quality assessment (0-100 scale)

---

## 📊 Batch Operations

### 11. Multiple Quotes

#### `get_multiple_quotes(symbols: List[str]) -> List[Quote]`

**Purpose**: Get quotes for multiple symbols efficiently

**Agent Example**:
```python
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
quotes = orchestrator.get_multiple_quotes(symbols)

print("📈 Portfolio Summary")
for quote in quotes:
    if quote and quote.price:
        change_emoji = "🟢" if quote.change_percent > 0 else "🔴"
        print(f"{change_emoji} {quote.symbol}: ${quote.price} ({quote.change_percent:+.2f}%)")
```

---

## 🎯 Performance and Quality Monitoring

### 12. Data Quality Reporting

#### `get_data_quality_report() -> Dict[str, DataQualityMetrics]`

**Purpose**: Get performance metrics for all data sources

**Returns**: Dictionary of DataQualityMetrics with:
- `source`: Data source name
- `quality_score`: Quality score 0-100
- `latency_ms`: Average response time
- `success_rate`: Success percentage
- `last_updated`: Last successful update
- `issues`: List of known issues

**Agent Example**:
```python
quality_report = orchestrator.get_data_quality_report()
print("🎯 Data Source Quality Report")
for source, metrics in quality_report.items():
    status_emoji = "🟢" if metrics.success_rate > 90 else "🟡" if metrics.success_rate > 70 else "🔴"
    print(f"{status_emoji} {source}: {metrics.quality_score:.1f}/100 quality, {metrics.latency_ms:.0f}ms avg")
    print(f"   ✅ Success Rate: {metrics.success_rate:.1f}%")
    if metrics.issues:
        print(f"   ⚠️  Issues: {', '.join(metrics.issues[:2])}")
```

---

## 🔧 Agent Integration Patterns

### Smart Quote Retrieval with Fallback

```python
def get_reliable_quote(symbol: str, max_retries: int = 3) -> Optional[Quote]:
    """Agent pattern: Reliable quote with quality validation"""

    for attempt in range(max_retries):
        quote = orchestrator.get_quote(symbol)

        if quote and quote.quality_score and quote.quality_score >= 80:
            return quote

        if attempt < max_retries - 1:
            print(f"⚠️  Quality score {quote.quality_score}/100, retrying...")
            time.sleep(1)

    return quote  # Return best available even if quality is low

# Usage
quote = get_reliable_quote("AAPL")
```

### Multi-Source Sentiment Analysis

```python
def analyze_market_sentiment(symbol: str) -> dict:
    """Agent pattern: Comprehensive sentiment analysis"""

    sentiment_data = orchestrator.get_sentiment_data(symbol)

    # Calculate weighted average sentiment
    total_weight = 0
    weighted_sentiment = 0

    source_weights = {"reddit": 0.4, "twitter": 0.4, "yahoo_news": 0.2}

    for source, sentiment in sentiment_data.items():
        weight = source_weights.get(source, 0.1) * sentiment.confidence
        weighted_sentiment += sentiment.sentiment_score * weight
        total_weight += weight

    if total_weight > 0:
        weighted_sentiment /= total_weight

    return {
        "symbol": symbol,
        "weighted_sentiment": weighted_sentiment,
        "confidence": total_weight / sum(source_weights.values()),
        "sources": {source: {"score": s.sentiment_score, "confidence": s.confidence}
                   for source, s in sentiment_data.items()},
        "trend": "bullish" if weighted_sentiment > 0.1 else "bearish" if weighted_sentiment < -0.1 else "neutral"
    }

# Usage
analysis = analyze_market_sentiment("AAPL")
print(f"🎯 {analysis['symbol']} Sentiment: {analysis['weighted_sentiment']:+.3f} ({analysis['trend']})")
```

### Market Overview Dashboard

```python
def create_market_overview() -> dict:
    """Agent pattern: Comprehensive market overview"""

    # Get major indices
    indices = ["SPY", "QQQ", "IWM", "VTI"]
    index_quotes = orchestrator.get_multiple_quotes(indices)

    # Get market breadth
    breadth = orchestrator.get_market_breadth()

    # Get sector performance
    sectors = orchestrator.get_sector_performance()

    # Get sentiment for major stocks
    major_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    sentiment_overview = {}
    for stock in major_stocks:
        sentiment_data = orchestrator.get_sentiment_data(stock)
        if sentiment_data:
            avg_sentiment = sum(s.sentiment_score for s in sentiment_data.values()) / len(sentiment_data)
            sentiment_overview[stock] = avg_sentiment

    return {
        "indices": {q.symbol: {"price": float(q.price), "change_pct": float(q.change_percent)}
                   for q in index_quotes if q},
        "breadth": {
            "advancers": breadth.advancers,
            "decliners": breadth.decliners,
            "ad_ratio": breadth.advance_decline_ratio
        } if breadth else None,
        "top_sectors": sorted(sectors, key=lambda x: x.performance_1d, reverse=True)[:3],
        "sentiment_leaders": dict(sorted(sentiment_overview.items(), key=lambda x: x[1], reverse=True))
    }

# Usage
overview = create_market_overview()
```

---

## ⚡ Performance Optimization Tips

### 1. Use Caching Effectively
- Reddit sentiment has 5-minute TTL caching (469,732x speedup for cached data)
- YFinance data caches internally for rapid retrieval
- SQLite-backed cache service provides persistent caching

### 2. Prefer Specific Timeframes
```python
# More efficient - specific period
data = orchestrator.get_market_data("AAPL", period="1mo", interval="1d")

# Less efficient - max period
data = orchestrator.get_market_data("AAPL", period="max", interval="1d")
```

### 3. Batch Operations
```python
# More efficient - single batch call
quotes = orchestrator.get_multiple_quotes(["AAPL", "MSFT", "GOOGL"])

# Less efficient - multiple individual calls
quotes = [orchestrator.get_quote(symbol) for symbol in ["AAPL", "MSFT", "GOOGL"]]
```

### 4. Quality-Based Source Selection
```python
# Monitor quality and switch sources if needed
quality_report = orchestrator.get_data_quality_report()
best_source = max(quality_report.items(), key=lambda x: x[1].quality_score)
print(f"Best performing source: {best_source[0]} ({best_source[1].quality_score:.1f}/100)")
```

---

## 🚨 Error Handling and Resilience

### Graceful Degradation Pattern

```python
def get_stock_analysis(symbol: str) -> dict:
    """Agent pattern: Resilient stock analysis with graceful degradation"""

    analysis = {"symbol": symbol, "status": "partial", "data": {}}

    # Try to get quote
    try:
        quote = orchestrator.get_quote(symbol)
        if quote:
            analysis["data"]["quote"] = {
                "price": float(quote.price),
                "change_percent": float(quote.change_percent),
                "quality_score": quote.quality_score
            }
            analysis["status"] = "success"
    except Exception as e:
        analysis["errors"] = analysis.get("errors", [])
        analysis["errors"].append(f"Quote error: {str(e)}")

    # Try to get sentiment (non-blocking)
    try:
        sentiment_data = orchestrator.get_sentiment_data(symbol)
        if sentiment_data:
            analysis["data"]["sentiment"] = {
                source: {"score": s.sentiment_score, "confidence": s.confidence}
                for source, s in sentiment_data.items()
            }
    except Exception as e:
        analysis["errors"] = analysis.get("errors", [])
        analysis["errors"].append(f"Sentiment error: {str(e)}")

    # Try to get market data (non-blocking)
    try:
        market_data = orchestrator.get_market_data(symbol, period="5d")
        if market_data:
            df = market_data.data
            analysis["data"]["market_data"] = {
                "latest_close": float(df['Close'].iloc[-1]),
                "5d_change": float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100),
                "avg_volume": float(df['Volume'].mean())
            }
    except Exception as e:
        analysis["errors"] = analysis.get("errors", [])
        analysis["errors"].append(f"Market data error: {str(e)}")

    return analysis

# Usage
analysis = get_stock_analysis("AAPL")
if analysis["status"] == "success":
    print(f"✅ Complete analysis for {analysis['symbol']}")
else:
    print(f"⚠️  Partial analysis for {analysis['symbol']}: {len(analysis['data'])} components")
```

---

## 📋 Available Data Sources

### Primary Sources (High Reliability)
- **YFinance**: Quotes, market data, company info (Quality: 100/100)
- **TwelveData**: Quotes, market data, real-time feeds (Quality: 80/100)
- **Financial Modeling Prep (FMP)**: Quotes, fundamentals, financial statements

### Sentiment Sources (Multi-source)
- **Reddit**: 8+ ticker mentions, FinBERT analysis (Cached: 5min TTL)
- **Twitter**: FinBERT-enhanced sentiment, 20+ samples (Quality: High confidence)
- **Yahoo News**: News-based sentiment analysis (Quality: 85% confidence)

### Market Analytics
- **FinViz**: Market breadth, sector performance (Quality: Real-time)
- **Investiny**: Historical data, international markets (Quality: 22 days OHLCV)

### Performance Characteristics
| Source | Avg Response Time | Quality Score | Reliability |
|--------|------------------|---------------|-------------|
| YFinance | 150-200ms | 100/100 | 99%+ |
| TwelveData | 200-400ms | 80/100 | 95%+ |
| Reddit Sentiment | 2.1s (0.001s cached) | Variable | 95%+ |
| Twitter Sentiment | 1.8s | High confidence | 90%+ |
| FinViz Analytics | 135-305ms | Real-time | 98%+ |

---

## 🎯 Current System Status

**Overall Performance**: 🟢 Excellent
**Test Success Rate**: ✅ 100% (24/24 tests passing)
**Average Response Times**:
- Quote retrieval: 384ms
- Market data: 595ms
- Sentiment analysis: 4.1s (multi-source)
- Market breadth: 305ms
- Sector performance: 136ms

**Quality Metrics**:
- YFinance market data: 100/100 quality score
- TwelveData quotes: 80/100 quality score
- Multi-source sentiment: High confidence scoring
- Intelligent caching: 30% performance improvement

**Data Freshness**:
- Real-time quotes: Sub-second latency
- Market data: End-of-day with intraday options
- Sentiment analysis: 5-minute refresh cycles
- Market breadth: Real-time market hours

---

This comprehensive guide provides agents with everything needed to effectively use Oracle-X's DataFeedOrchestrator for robust financial data analysis and decision-making.
