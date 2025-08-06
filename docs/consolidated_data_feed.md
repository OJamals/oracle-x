# Consolidated Financial Data Feed

A comprehensive financial data aggregator that unifies multiple data sources with intelligent fallback, caching, and rate limiting.

## Features

- **Unified Interface**: Single API for all financial data needs
- **Intelligent Fallback**: Automatic switching between data sources if one fails
- **Rate Limiting**: Respects API limits to prevent quota exhaustion  
- **Caching**: Reduces API calls with intelligent TTL-based caching
- **Data Standardization**: Consistent data formats across all sources
- **Comprehensive Coverage**: Quotes, historical data, financials, news, and more

## Supported Data Sources

### Tier 1 (Free, Unlimited)
- **yfinance**: Primary source for most data
- **financedatabase**: Static reference data

### Tier 2 (Free, Limited)
- **finnhub**: 60 calls/minute, excellent for news and company data
- **Financial Modeling Prep**: 250 calls/day, comprehensive financial statements
- **Alpha Vantage**: 5 calls/minute, good for technical indicators

### Tier 3 (Experimental/Limited)
- **investiny**: Web scraping based
- **quantsumore**: Specialized quant data
- **stockdex**: Alternative data source

## API Rate Limits Summary

| Source | Free Tier Limit | Best Use Case |
|--------|----------------|---------------|
| yfinance | Unlimited | Primary for quotes, historical, company info |
| finnhub | 60/minute | News, company profiles, market data |
| FMP | 250/day | Financial statements, detailed fundamentals |
| Alpha Vantage | 5/minute, 500/day | Technical indicators, news sentiment |
| financedatabase | Unlimited | Static security metadata |

## Quick Start

```python
from consolidated_data_feed import get_quote, get_historical, get_company_info

# Get real-time quote
quote = get_quote("AAPL")
print(f"AAPL: ${quote.price} ({quote.change:+.2f}%)")

# Get historical data
hist = get_historical("AAPL", period="1y")
print(f"Got {len(hist)} days of data")

# Get company information
info = get_company_info("AAPL")
print(f"{info.name} - {info.sector}")
```

## Advanced Usage

```python
from consolidated_data_feed import ConsolidatedDataFeed

feed = ConsolidatedDataFeed()

# Multiple quotes efficiently
quotes = feed.get_multiple_quotes(["AAPL", "GOOGL", "MSFT"])

# Company news
news = feed.get_news("AAPL", limit=5)

# Search securities
results = feed.search_securities(country="United States", sector="Technology")

# Cache management
stats = feed.get_cache_stats()
feed.clear_cache()
```

## Available Data Types

### Real-time Market Data
- `get_quote(symbol)` - Current price, volume, market cap
- `get_multiple_quotes(symbols)` - Bulk quotes

### Historical Data
- `get_historical(symbol, period)` - OHLCV data
- Supports periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

### Company Information
- `get_company_info(symbol)` - Profile, sector, industry, description
- Financial metrics and ratios
- Executive information

### News & Sentiment
- `get_news(symbol, limit)` - Recent company news
- Multi-source aggregation
- Sentiment scoring (where available)

### Financial Statements
- Income statements (annual/quarterly)
- Balance sheets
- Cash flow statements
- Financial ratios and metrics

## Data Quality & Reliability

### Source Selection Logic
1. **yfinance first** - Free, reliable, comprehensive
2. **Fallback to paid APIs** - When yfinance fails or data unavailable
3. **Specialized sources** - For specific data types (e.g., finnhub for news)

### Data Validation
- Schema validation for all responses
- Automatic data type conversion
- Outlier detection and filtering
- Cross-source data verification

### Error Handling
- Graceful degradation on source failures
- Exponential backoff for rate limiting
- Circuit breaker pattern for failing sources
- Comprehensive error logging

## Caching Strategy

| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| Real-time quotes | 30 seconds | Balance freshness vs API usage |
| Daily historical | 1 hour | Static after market close |
| Intraday historical | 5 minutes | More frequent updates needed |
| Financial statements | 24 hours | Updated quarterly |
| Company info | 7 days | Rarely changes |
| News | 30 minutes | Regular refresh for updates |

## Configuration

The system uses `config/data_feed_config.yaml` for:
- Source priorities per data type
- Rate limiting settings
- Cache TTL configuration
- API timeout settings
- Data quality filters

## Environment Variables Required

```bash
# Required for paid API sources
FINNHUB_API_KEY=your_finnhub_key
FINANCIALMODELINGPREP_API_KEY=your_fmp_key
ALPHAVANTAGE_API_KEY=your_av_key

# Optional for enhanced features
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
```

## Performance Optimization

### Batch Operations
- Use `get_multiple_quotes()` for bulk quote requests
- Implement connection pooling for HTTP requests
- Async support for high-throughput scenarios

### Memory Management
- LRU cache eviction for memory efficiency
- Configurable cache size limits
- Periodic cache cleanup

### Network Optimization
- Request compression when supported
- Connection reuse
- Intelligent request batching

## Integration Examples

### Oracle-X Signal Generation
```python
# Get data for signal generation
feed = ConsolidatedDataFeed()
quote = feed.get_quote("AAPL")
hist = feed.get_historical("AAPL", period="3mo")
news = feed.get_news("AAPL", limit=10)

# Use in trading strategy
if quote.price > hist['Close'].rolling(50).mean().iloc[-1]:
    # Price above 50-day MA, consider bullish signal
    pass
```

### Portfolio Analytics
```python
# Get portfolio data
portfolio = ["AAPL", "GOOGL", "MSFT", "TSLA"]
quotes = feed.get_multiple_quotes(portfolio)

# Calculate portfolio value
total_value = sum(quote.price * shares for symbol, quote in quotes.items())
```

### Risk Management
```python
# Monitor portfolio risk
for symbol in portfolio:
    quote = feed.get_quote(symbol)
    if abs(quote.change_percent) > 5.0:
        # Alert on large moves
        send_alert(f"{symbol} moved {quote.change_percent:.1f}%")
```

## Troubleshooting

### Common Issues
1. **API Key Issues**: Verify all API keys are set in `.env`
2. **Rate Limiting**: Monitor rate limit logs, implement delays if needed
3. **Data Quality**: Check source priority if getting unexpected data
4. **Cache Issues**: Clear cache if seeing stale data

### Debugging
```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Check cache stats
print(feed.get_cache_stats())

# Force cache clear
feed.clear_cache()
```

### Monitoring
- Monitor API call counts per source
- Track cache hit rates
- Monitor response times
- Set up alerts for API failures

## Future Enhancements

### Planned Features
- WebSocket real-time data streams
- Cryptocurrency data integration
- Options chain analysis
- ESG and sustainability metrics
- Alternative data sources (satellite, social media)

### Performance Improvements
- Async/await support for better concurrency
- Database backend for persistent caching
- GraphQL API interface
- Real-time data subscriptions
