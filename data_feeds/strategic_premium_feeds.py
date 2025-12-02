"""
Strategic Premium API Integration
Uses Finnhub and FMP ONLY for advanced data points not available from free sources.

Free Sources (yfinance, web scraping) provide:
- Basic quotes (price, volume, change)
- Historical prices
- Options chains
- Company info
- Income statements
- Balance sheets
- Cash flow statements

Premium APIs provide UNIQUE data:
- Finnhub: Insider trading signals, recommendation trends, price targets, social sentiment
- FMP: Analyst estimates, institutional ownership changes, insider transactions, DCF valuations
"""

import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import logging

try:  # Ensure local .env values are available when module is imported directly
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from core.config import config as _core_config
except Exception:
    _core_config = None

logger = logging.getLogger(__name__)

# Rate limiting configuration
class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = deque(maxlen=calls_per_minute)
    
    def wait_if_needed(self):
        """Wait if we've exceeded rate limit"""
        now = time.time()
        
        # Remove calls older than 1 minute
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        
        # If at limit, wait
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        # Record this call
        self.calls.append(now)


class StrategicPremiumFeeds:
    """
    Strategic use of premium APIs for unique data only.
    Implements batching and rate limiting.
    """
    
    def __init__(self):
        # Initialize rate limiters
        self.finnhub_limiter = RateLimiter(calls_per_minute=50)  # Conservative limit
        self.fmp_limiter = RateLimiter(calls_per_minute=200)  # Conservative for free tier
        
        # Check API keys
        config_data = getattr(_core_config, "data_feeds", None)

        self.finnhub_key = os.environ.get("FINNHUB_API_KEY") or getattr(config_data, "finnhub_api_key", None)
        self.fmp_key = (
            os.environ.get("FINANCIALMODELINGPREP_API_KEY")
            or os.environ.get("FMP_API_KEY")
            or getattr(config_data, "financial_modeling_prep_api_key", None)
        )
        
        self.finnhub_available = bool(self.finnhub_key)
        self.fmp_available = bool(self.fmp_key)
        
        if not self.finnhub_available:
            logger.warning("Finnhub API key not found - insider signals disabled")
        if not self.fmp_available:
            logger.warning("FMP API key not found - analyst estimates disabled")
    
    def get_unique_finnhub_data(self, symbols: List[str], max_symbols: int = 5) -> Dict[str, Any]:
        """
        Get UNIQUE data from Finnhub not available in free sources.
        
        Unique Data Points:
        - Insider sentiment (buy/sell by executives)
        - Recommendation trends (analyst upgrades/downgrades)  
        - Price targets from analysts
        - Social sentiment scores
        
        Args:
            symbols: List of ticker symbols
            max_symbols: Maximum symbols to process (rate limiting)
        
        Returns:
            Dict with unique Finnhub data
        """
        if not self.finnhub_available:
            return {"available": False, "reason": "No API key"}
        
        import requests
        
        result = {
            "available": True,
            "insider_sentiment": {},
            "recommendation_trends": {},
            "price_targets": {},
            "fetch_errors": []
        }
        
        # Limit symbols to avoid rate limits
        symbols = symbols[:max_symbols]
        
        for symbol in symbols:
            try:
                # Insider Sentiment (UNIQUE to Finnhub)
                self.finnhub_limiter.wait_if_needed()
                url = f"https://finnhub.io/api/v1/stock/insider-sentiment"
                params = {
                    "symbol": symbol,
                    "from": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                    "to": datetime.now().strftime("%Y-%m-%d"),
                    "token": self.finnhub_key
                }
                resp = requests.get(url, params=params, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data and "data" in data and data["data"]:
                        # Calculate net insider sentiment
                        total_buy = sum(d.get("change", 0) for d in data["data"] if d.get("change", 0) > 0)
                        total_sell = sum(abs(d.get("change", 0)) for d in data["data"] if d.get("change", 0) < 0)
                        
                        result["insider_sentiment"][symbol] = {
                            "net_buying": total_buy - total_sell,
                            "transactions": len(data["data"]),
                            "sentiment": "bullish" if total_buy > total_sell else "bearish" if total_sell > total_buy else "neutral"
                        }
                
                # Recommendation Trends (UNIQUE aggregation)
                self.finnhub_limiter.wait_if_needed()
                url = f"https://finnhub.io/api/v1/stock/recommendation"
                params = {"symbol": symbol, "token": self.finnhub_key}
                resp = requests.get(url, params=params, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) > 0:
                        latest = data[0]
                        result["recommendation_trends"][symbol] = {
                            "buy": latest.get("buy", 0),
                            "hold": latest.get("hold", 0),
                            "sell": latest.get("sell", 0),
                            "strong_buy": latest.get("strongBuy", 0),
                            "strong_sell": latest.get("strongSell", 0),
                            "consensus": self._calculate_consensus(latest)
                        }
                
                # Price Targets (UNIQUE)
                self.finnhub_limiter.wait_if_needed()
                url = f"https://finnhub.io/api/v1/stock/price-target"
                params = {"symbol": symbol, "token": self.finnhub_key}
                resp = requests.get(url, params=params, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data and "targetMean" in data:
                        result["price_targets"][symbol] = {
                            "target_high": data.get("targetHigh"),
                            "target_low": data.get("targetLow"),
                            "target_mean": data.get("targetMean"),
                            "target_median": data.get("targetMedian"),
                            "num_analysts": data.get("numberOfAnalysts", 0)
                        }
                
            except Exception as e:
                result["fetch_errors"].append(f"{symbol}: {str(e)}")
                logger.error(f"Finnhub error for {symbol}: {e}")
        
        return result
    
    def get_unique_fmp_data(self, symbols: List[str], max_symbols: int = 5) -> Dict[str, Any]:
        """
        Get UNIQUE data from FMP not available in free sources.
        
        Unique Data Points:
        - Analyst earnings/revenue estimates (forward-looking)
        - Institutional ownership changes (smart money tracking)
        - Insider transaction details (detailed buys/sells)
        - DCF valuations (fair value estimates)
        
        Args:
            symbols: List of ticker symbols
            max_symbols: Maximum symbols to process (rate limiting)
        
        Returns:
            Dict with unique FMP data
        """
        if not self.fmp_available:
            return {"available": False, "reason": "No API key"}
        
        import requests
        
        result = {
            "available": True,
            "analyst_estimates": {},
            "institutional_ownership": {},
            "insider_transactions": {},
            "dcf_valuations": {},
            "fetch_errors": []
        }
        
        # Limit symbols to avoid rate limits
        symbols = symbols[:max_symbols]
        
        for symbol in symbols:
            try:
                # Analyst Estimates (UNIQUE forward-looking data)
                self.fmp_limiter.wait_if_needed()
                url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}"
                params = {"apikey": self.fmp_key, "limit": 4}
                resp = requests.get(url, params=params, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) > 0:
                        latest = data[0]
                        result["analyst_estimates"][symbol] = {
                            "estimated_eps": latest.get("estimatedEpsAvg"),
                            "estimated_eps_high": latest.get("estimatedEpsHigh"),
                            "estimated_eps_low": latest.get("estimatedEpsLow"),
                            "estimated_revenue": latest.get("estimatedRevenueAvg"),
                            "num_analysts": latest.get("numberAnalystEstimatedRevenue", 0),
                            "date": latest.get("date")
                        }
                
                # Institutional Ownership (UNIQUE - shows smart money)
                self.fmp_limiter.wait_if_needed()
                url = f"https://financialmodelingprep.com/api/v3/institutional-holder/{symbol}"
                params = {"apikey": self.fmp_key}
                resp = requests.get(url, params=params, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) > 0:
                        # Get top 5 institutional holders
                        top_holders = data[:5]
                        total_shares = sum(h.get("shares", 0) for h in top_holders)
                        result["institutional_ownership"][symbol] = {
                            "top_5_holders": [h.get("holder") for h in top_holders],
                            "top_5_shares": total_shares,
                            "total_holders": len(data)
                        }
                
                # Insider Transactions (UNIQUE detailed data)
                self.fmp_limiter.wait_if_needed()
                url = f"https://financialmodelingprep.com/api/v4/insider-trading"
                params = {"symbol": symbol, "apikey": self.fmp_key, "limit": 10}
                resp = requests.get(url, params=params, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) > 0:
                        # Calculate recent insider activity
                        buys = [t for t in data if t.get("transactionType", "").lower() in ["p-purchase", "m-exempt"]]
                        sells = [t for t in data if t.get("transactionType", "").lower() in ["s-sale"]]
                        
                        result["insider_transactions"][symbol] = {
                            "recent_buys": len(buys),
                            "recent_sells": len(sells),
                            "net_sentiment": "bullish" if len(buys) > len(sells) else "bearish" if len(sells) > len(buys) else "neutral",
                            "total_transactions": len(data)
                        }
                
                # DCF Valuation (UNIQUE fair value estimate)
                self.fmp_limiter.wait_if_needed()
                url = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/{symbol}"
                params = {"apikey": self.fmp_key}
                resp = requests.get(url, params=params, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) > 0:
                        dcf_data = data[0]
                        result["dcf_valuations"][symbol] = {
                            "dcf_value": dcf_data.get("dcf"),
                            "stock_price": dcf_data.get("Stock Price"),
                            "date": dcf_data.get("date")
                        }
                
            except Exception as e:
                result["fetch_errors"].append(f"{symbol}: {str(e)}")
                logger.error(f"FMP error for {symbol}: {e}")
        
        return result
    
    def get_all_premium_data(self, symbols: List[str], max_symbols: int = 5) -> Dict[str, Any]:
        """
        Get all unique premium data for symbols with smart batching.
        
        Args:
            symbols: List of ticker symbols
            max_symbols: Maximum symbols to process (rate limiting)
        
        Returns:
            Combined dict with all unique premium data
        """
        logger.info(f"Fetching premium data for {min(len(symbols), max_symbols)} symbols")
        
        result = {
            "finnhub": self.get_unique_finnhub_data(symbols, max_symbols),
            "fmp": self.get_unique_fmp_data(symbols, max_symbols),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_consensus(self, rec_data: Dict) -> str:
        """Calculate analyst consensus from recommendation data"""
        buy = rec_data.get("buy", 0) + rec_data.get("strongBuy", 0) * 2
        sell = rec_data.get("sell", 0) + rec_data.get("strongSell", 0) * 2
        hold = rec_data.get("hold", 0)
        
        if buy > sell + hold:
            return "strong_buy"
        elif buy > sell:
            return "buy"
        elif sell > buy + hold:
            return "strong_sell"
        elif sell > buy:
            return "sell"
        else:
            return "hold"


# Singleton instance
_premium_feeds = None

def get_premium_feeds() -> StrategicPremiumFeeds:
    """Get singleton premium feeds instance"""
    global _premium_feeds
    if _premium_feeds is None:
        _premium_feeds = StrategicPremiumFeeds()
    return _premium_feeds


def fetch_premium_unique_data(symbols: List[str], max_symbols: int = 5) -> Dict[str, Any]:
    """Convenience wrapper used by sync/async pipelines to fetch premium-only signals."""
    premium = get_premium_feeds()

    # Bail out quickly if no premium APIs configured to avoid repeated warnings upstream
    if not (premium.finnhub_available or premium.fmp_available):
        return {
            "available": False,
            "reason": "No premium API keys configured",
            "finnhub": {"available": False, "reason": "No API key"},
            "fmp": {"available": False, "reason": "No API key"},
        }

    try:
        selected = symbols[:max_symbols] if symbols else []
        return {
            "available": True,
            "finnhub": premium.get_unique_finnhub_data(selected, max_symbols=max_symbols),
            "fmp": premium.get_unique_fmp_data(selected, max_symbols=max_symbols),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:  # Defensive catch to preserve upstream pipeline stability
        logger.warning(f"Premium unique data fetch failed: {exc}")
        return {
            "available": False,
            "reason": str(exc),
            "finnhub": {"available": bool(premium.finnhub_available)},
            "fmp": {"available": bool(premium.fmp_available)},
        }
