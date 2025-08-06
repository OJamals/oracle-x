"""
Simplified Oracle-X Data Feeds Module
Uses the Consolidated Data Feed as the backend for all financial data.
This module provides backward compatibility with existing Oracle-X code
while leveraging the new unified data infrastructure.

Change note: Removed sys.path manipulation and switched to absolute imports from data_feeds.consolidated_data_feed to avoid ambiguity.
"""

# Route legacy facade through DataFeedOrchestrator while preserving shapes
from data_feeds.data_feed_orchestrator import (
    get_quote as orch_get_quote,
    get_market_data as orch_get_market_data,
    get_company_info as orch_get_company_info,
    get_news as orch_get_news,
    get_multiple_quotes as orch_get_multiple_quotes,
)
# Keep types from consolidated for legacy type hints
from data_feeds.consolidated_data_feed import (
    ConsolidatedDataFeed,
    Quote,
    CompanyInfo,
    NewsItem,
    get_historical as _legacy_get_historical,  # only for internal provider delegation
)
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Global data feed instance
_global_feed = None

def get_data_feed():
    """Get the global consolidated data feed instance"""
    global _global_feed
    if _global_feed is None:
        _global_feed = ConsolidatedDataFeed()
    return _global_feed

class UnifiedDataProvider:
    """
    Unified data provider that replaces all individual data feed modules.
    Provides backward compatibility with existing Oracle-X interfaces.
    """
    
    def __init__(self):
        self.feed = get_data_feed()
        logger.info("UnifiedDataProvider initialized with ConsolidatedDataFeed")
    
    # Market Data Methods
    def get_stock_price(self, symbol: str) -> Optional[float]:
        """Get current stock price (legacy interface)"""
        # Delegate via orchestrator module-level forward to keep caching/selection centralized
        quote_obj = orch_get_quote(symbol)
        return float(quote_obj.price) if quote_obj else None
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical stock data (legacy interface)"""
        # Use orchestrator market data and return the normalized DataFrame
        md = orch_get_market_data(symbol, period, "1d")
        if md is None:
            return None
        # Orchestrator returns MarketData with df in .data; preserve legacy return as DataFrame
        return md.data
    
    def get_real_time_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote as dictionary (legacy interface)"""
        quote = orch_get_quote(symbol)
        if not quote:
            return None
        return {
            'symbol': quote.symbol,
            'price': float(quote.price),
            'change': float(quote.change),
            'change_percent': float(quote.change_percent),
            'volume': quote.volume,
            'market_cap': quote.market_cap,
            'source': quote.source
        }
    
    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get multiple quotes efficiently"""
        quotes_map = orch_get_multiple_quotes(symbols)
        result = {}
        for symbol, quote in quotes_map.items():
            result[symbol] = {
                'symbol': quote.symbol,
                'price': float(quote.price),
                'change': float(quote.change),
                'change_percent': float(quote.change_percent),
                'volume': quote.volume,
                'market_cap': quote.market_cap,
                'source': quote.source
            }
        return result
    
    # Company Information
    def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """Get company profile (legacy interface)"""
        info = orch_get_company_info(symbol)
        if info:
            return {
                'symbol': info.symbol,
                'name': info.name,
                'sector': info.sector,
                'industry': info.industry,
                'country': info.country,
                'exchange': info.exchange,
                'market_cap': info.market_cap,
                'employees': info.employees,
                'description': info.description,
                'website': info.website,
                'source': info.source
            }
        return None
    
    # News and Sentiment
    def get_company_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get company news (legacy interface)"""
        news = orch_get_news(symbol, limit)
        result = []
        for item in news:
            result.append({
                'title': item.title,
                'summary': item.summary,
                'url': item.url,
                'published': item.published.isoformat(),
                'source': item.source
            })
        return result
    
    # Technical Analysis Support
    def get_ohlcv_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get OHLCV data for technical analysis"""
        md = orch_get_market_data(symbol, period, "1d")
        return md.data if md is not None else None
    
    def calculate_technical_indicators(self, symbol: str, period: str = "6mo") -> Optional[Dict]:
        """Calculate common technical indicators"""
        hist = self.feed.get_historical(symbol, period)
        if hist is None or hist.empty:
            return None
        
        # Calculate indicators
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = hist['Close'].rolling(window=20).mean().iloc[-1]
        indicators['sma_50'] = hist['Close'].rolling(window=50).mean().iloc[-1]
        indicators['ema_12'] = hist['Close'].ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = hist['Close'].ewm(span=26).mean().iloc[-1]
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(hist['Close']).iloc[-1]
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(hist['Close'])
        indicators['bb_upper'] = bb_upper.iloc[-1]
        indicators['bb_lower'] = bb_lower.iloc[-1]
        
        # MACD
        macd_line, signal_line = self._calculate_macd(hist['Close'])
        indicators['macd'] = macd_line.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        
        # Volume indicators
        indicators['volume_sma_20'] = hist['Volume'].rolling(window=20).mean().iloc[-1]
        
        return indicators
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line
    
    # Portfolio Analysis
    def get_portfolio_data(self, holdings: Dict[str, float]) -> Dict:
        """
        Get comprehensive portfolio data
        holdings: dict of symbol -> quantity
        """
        symbols = list(holdings.keys())
        quotes = self.feed.get_multiple_quotes(symbols)
        
        portfolio_value = 0
        portfolio_change = 0
        positions = {}
        
        for symbol, quantity in holdings.items():
            if symbol in quotes:
                quote = quotes[symbol]
                position_value = float(quote.price) * quantity
                position_change = float(quote.change) * quantity
                
                portfolio_value += position_value
                portfolio_change += position_change
                
                positions[symbol] = {
                    'quantity': quantity,
                    'price': float(quote.price),
                    'value': position_value,
                    'change': position_change,
                    'change_percent': float(quote.change_percent),
                    'weight': 0  # Will be calculated after total value is known
                }
        
        # Calculate position weights
        for symbol in positions:
            positions[symbol]['weight'] = positions[symbol]['value'] / portfolio_value if portfolio_value > 0 else 0
        
        return {
            'total_value': portfolio_value,
            'total_change': portfolio_change,
            'change_percent': (portfolio_change / (portfolio_value - portfolio_change)) * 100 if portfolio_value > portfolio_change else 0,
            'positions': positions,
            'symbols': list(positions.keys()),
            'position_count': len(positions)
        }
    
    # Risk Management
    def monitor_portfolio_risk(self, holdings: Dict[str, float], 
                             volatility_threshold: float = 5.0) -> Dict:
        """Monitor portfolio for risk alerts"""
        portfolio_data = self.get_portfolio_data(holdings)
        alerts = []
        
        for symbol, position in portfolio_data['positions'].items():
            change_pct = abs(position['change_percent'])
            
            if change_pct > volatility_threshold:
                alert_level = "HIGH" if change_pct > 10 else "MODERATE"
                alerts.append({
                    'symbol': symbol,
                    'level': alert_level,
                    'change_percent': position['change_percent'],
                    'message': f"{symbol} moved {position['change_percent']:+.1f}%"
                })
        
        return {
            'alerts': alerts,
            'high_risk_count': len([a for a in alerts if a['level'] == "HIGH"]),
            'moderate_risk_count': len([a for a in alerts if a['level'] == "MODERATE"]),
            'portfolio_data': portfolio_data
        }
    
    # Cache Management
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        return self.feed.get_cache_stats()
    
    def clear_cache(self):
        """Clear all cached data"""
        self.feed.clear_cache()

# Backward compatibility instances
data_provider = UnifiedDataProvider()

# Legacy function interfaces for backward compatibility
def get_stock_price(symbol: str) -> Optional[float]:
    return data_provider.get_stock_price(symbol)

def get_stock_data(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    return data_provider.get_stock_data(symbol, period)

def get_company_profile(symbol: str) -> Optional[Dict]:
    return data_provider.get_company_profile(symbol)

def get_company_news(symbol: str, limit: int = 10) -> List[Dict]:
    return data_provider.get_company_news(symbol, limit)

def get_real_time_quote(symbol: str) -> Optional[Dict]:
    return data_provider.get_real_time_quote(symbol)

def get_multiple_quotes(symbols: List[str]) -> Dict[str, Dict]:
    return data_provider.get_multiple_quotes(symbols)

# Export main classes and functions
__all__ = [
    'UnifiedDataProvider',
    'data_provider',
    'get_stock_price',
    'get_stock_data', 
    'get_company_profile',
    'get_company_news',
    'get_real_time_quote',
    'get_multiple_quotes',
    'get_data_feed'
]
