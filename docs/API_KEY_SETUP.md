# 🔑 API Key Configuration Guide

This guide provides comprehensive instructions for configuring all required API keys to enable full functionality of the ORACLE-X trading system.

## 📋 Quick Start

1. **Copy the template**: `cp .env.example .env`
2. **Get API keys** from the providers listed below
3. **Configure keys** in your `.env` file
4. **Validate setup**: `python scripts/validate_api_keys.py`
5. **Test functionality**: Verify all adapters work correctly

## 🔧 Required API Keys

### Financial Data APIs

#### 1. TwelveData
- **Purpose**: High-quality market data, technical indicators, and fundamental data
- **Environment Variable**: `TWELVEDATA_API_KEY`
- **Website**: [TwelveData](https://twelvedata.com/)
- **Pricing**: Free tier available, paid plans for higher limits
- **Setup**:
  1. Sign up for a TwelveData account
  2. Navigate to API section in dashboard
  3. Generate your API key
  4. Add to `.env`: `TWELVEDATA_API_KEY=your_key_here`

#### 2. Financial Modeling Prep (FMP)
- **Purpose**: Comprehensive financial statements, ratios, and market data
- **Environment Variable**: `FINANCIALMODELINGPREP_API_KEY`
- **Website**: [Financial Modeling Prep](https://financialmodelingprep.com/)
- **Pricing**: Free tier (250 requests/day), paid plans available
- **Setup**:
  1. Register at Financial Modeling Prep
  2. Verify email to get API key
  3. Add to `.env`: `FINANCIALMODELINGPREP_API_KEY=your_key_here`

#### 3. Finnhub
- **Purpose**: Real-time stock quotes, news, and market data
- **Environment Variable**: `FINNHUB_API_KEY`
- **Website**: [Finnhub](https://finnhub.io/)
- **Pricing**: Free tier (60 requests/minute), paid plans for higher limits
- **Setup**:
  1. Create Finnhub account
  2. Go to "Get Free API Key" section
  3. Copy your API key
  4. Add to `.env`: `FINNHUB_API_KEY=your_key_here`

#### 4. Alpha Vantage
- **Purpose**: Stock market data and fundamental analysis
- **Environment Variable**: `ALPHAVANTAGE_API_KEY`
- **Website**: [Alpha Vantage](https://www.alphavantage.co/)
- **Pricing**: Free tier (5 requests/minute), paid plans available
- **Setup**:
  1. Sign up for Alpha Vantage
  2. Check email for API key
  3. Add to `.env`: `ALPHAVANTAGE_API_KEY=your_key_here`

#### 5. Polygon.io
- **Purpose**: Advanced market data, options data, and analytics
- **Environment Variable**: `POLYGON_API_KEY`
- **Website**: [Polygon.io](https://polygon.io/)
- **Pricing**: Free tier (5 requests/minute), paid plans for real-time data
- **Setup**:
  1. Register at Polygon.io
  2. Choose a plan (free tier available)
  3. Get your API key from dashboard
  4. Add to `.env`: `POLYGON_API_KEY=your_key_here`

### Social Media & Sentiment APIs

#### 6. Twitter API (Bearer Token) - Optional
- **Purpose**: Enhanced sentiment analysis using Twitter data
- **Environment Variable**: `TWITTER_BEARER_TOKEN`
- **Website**: [Twitter Developer Portal](https://developer.twitter.com/)
- **Pricing**: Free tier (300 requests/15 minutes), paid plans available
- **Important Note**: The system uses `twscrape` for Twitter data collection, which doesn't require official Twitter API keys. This Bearer Token is optional and only needed if you want to use the official Twitter API instead of scraping.
- **Setup** (only if you want to use official Twitter API):
  1. Apply for Twitter Developer Account
  2. Create a new App in the Developer Portal
  3. Generate Bearer Token in App settings
  4. Add to `.env`: `TWITTER_BEARER_TOKEN=your_bearer_token_here`

#### 7. Reddit API
- **Purpose**: Additional sentiment data from Reddit discussions
- **Environment Variables**: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`
- **Website**: [Reddit Apps](https://www.reddit.com/prefs/apps)
- **Pricing**: Free
- **Setup**:
  1. Go to Reddit Apps preferences
  2. Click "Create App" (choose "script" type)
  3. Fill in required details
  4. Copy Client ID and Secret
  5. Add to `.env`:
     ```
     REDDIT_CLIENT_ID=your_client_id_here
     REDDIT_CLIENT_SECRET=your_client_secret_here
     REDDIT_USER_AGENT=your_app_name/1.0
     ```

## 📊 Optional API Keys (Enhanced Features)

### Additional Data Providers

#### IEX Cloud
- **Environment Variable**: `IEX_API_KEY`
- **Purpose**: Alternative stock market data
- **Website**: [IEX Cloud](https://iexcloud.io/)

#### TradingEconomics
- **Environment Variable**: `TRADINGECONOMICS_API_KEY`
- **Purpose**: Economic indicators and market analysis
- **Website**: [TradingEconomics](https://tradingeconomics.com/)

#### FRED (Federal Reserve Economic Data)
- **Environment Variable**: `FRED_API_KEY`
- **Purpose**: Economic data from Federal Reserve
- **Website**: [FRED API](https://fred.stlouisfed.org/docs/api/fred/)

### News APIs

#### NewsAPI
- **Environment Variable**: `NEWSAPI_API_KEY`
- **Purpose**: News articles and headlines
- **Website**: [NewsAPI](https://newsapi.org/)

#### Google News API
- **Environment Variable**: `GOOGLE_NEWS_API_KEY`
- **Purpose**: Google News integration
- **Website**: [Google Custom Search API](https://developers.google.com/custom-search/v1/overview)

## 🔍 Validation & Testing

### Validate Configuration
Run the validation script to ensure all keys are properly configured:

```bash
python scripts/validate_api_keys.py
```

This will provide a comprehensive report showing:
- ✅ Valid API keys
- ❌ Missing required keys
- ⚠️  Optional keys status
- 💡 Recommendations

### Test Individual Adapters
Test each data adapter to ensure it's working:

```bash
# Test all adapters
python scripts/test_all_data_feeds.py

# Test individual adapters
python -c "from data_feeds.finnhub import fetch_finnhub_quote; print(fetch_finnhub_quote('AAPL'))"
python -c "from data_feeds.consolidated_data_feed import FMPAdapter; fmp = FMPAdapter(); quote = fmp.get_quote('AAPL'); print(f'AAPL: ${quote.price}' if quote else 'Failed')"
```

### Expected Validation Results
After proper configuration, you should see:

```
📊 SUMMARY:
   Total API Keys: 8
   Valid Keys: 8
   Missing Keys: 0
   Required Missing: 0

✅ VALID API KEYS:
   ✓ TwelveData API Key
   ✓ Financial Modeling Prep API Key
   ✓ Finnhub API Key
   ✓ Alpha Vantage API Key
   ✓ Polygon.io API Key
   ✓ Twitter Bearer Token
   ✓ Reddit Client ID
   ✓ Reddit Client Secret
```

## 🚨 Troubleshooting

### Common Issues

#### 1. API Key Not Recognized
**Problem**: Validation shows key as missing despite being in `.env`
**Solutions**:
- Check for typos in environment variable names
- Ensure no extra spaces around the `=` sign
- Restart your application/shell to reload environment
- Verify the key format with the provider

#### 2. Rate Limiting Errors
**Problem**: Getting 429 (Too Many Requests) errors
**Solutions**:
- Check your API provider's rate limits
- Upgrade to paid plans for higher limits
- Implement request throttling in your code
- Use fallback mechanisms for rate-limited services

#### 3. Invalid API Key Format
**Problem**: API provider rejects the key format
**Solutions**:
- Verify the key format with the provider's documentation
- Check if the key has expired
- Regenerate the key from the provider's dashboard
- Ensure you're using the correct key type (e.g., Bearer Token vs API Key)

### Debug Commands

```bash
# Check environment variables
env | grep -E "(TWELVEDATA|FINNHUB|FINANCIALMODELINGPREP|ALPHAVANTAGE|POLYGON)" | sort

# Test API connectivity
curl -H "Authorization: Bearer YOUR_TWITTER_BEARER_TOKEN" "https://api.twitter.com/2/tweets/search/recent?query=stocks"

# Check Python environment
python -c "import os; print('TWELVEDATA_API_KEY:', os.getenv('TWELVEDATA_API_KEY')[:10] + '...' if os.getenv('TWELVEDATA_API_KEY') else 'Not set')"
```

## 🔐 Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** instead of hardcoding keys
3. **Rotate keys periodically** for security
4. **Use different keys** for development and production
5. **Monitor API usage** to detect unusual activity
6. **Implement rate limiting** to prevent abuse
7. **Use HTTPS** for all API communications

## 📈 Production Deployment Checklist

- [ ] All required API keys configured
- [ ] Validation script passes without errors
- [ ] All adapters tested successfully
- [ ] Rate limiting configured appropriately
- [ ] Error handling and fallbacks implemented
- [ ] Monitoring and logging set up
- [ ] API key rotation process documented

## 🔗 Useful Links

- [API Key Management Best Practices](https://www.twilio.com/blog/api-key-management-best-practices)
- [Managing API Keys Securely](https://cloud.google.com/docs/authentication/api-keys)
- [OAuth 2.0 Bearer Tokens](https://tools.ietf.org/html/rfc6750)

## 📞 Support

If you encounter issues with API key configuration:

1. Run the validation script: `python scripts/validate_api_keys.py`
2. Check the troubleshooting section above
3. Verify your API keys with each provider's dashboard
4. Review the provider-specific documentation
5. Contact the specific API provider for support

---

*Last updated: 2025-01-21*
*For questions about this guide, please refer to the main project documentation.*
