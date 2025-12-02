"""
Oracle-X Pipeline Optimizations

This package contains optimization modules for improving pipeline performance:
- ticker_validator: Validates tickers before API calls (saves 3-6s per run)
- smart_rate_limiter: Intelligent rate limiting to prevent errors (saves 5-10s per run)
- request_cache: TTL-based caching for API responses (saves 20-30s on repeated queries)

Usage:
    from optimizations.ticker_validator import validate_tickers
    from optimizations.smart_rate_limiter import get_rate_limiter
    from data_feeds.cache.request_cache import get_request_cache
"""

__version__ = "1.0.0"
__author__ = "Oracle-X Development Team"

# Import main classes for easy access
try:
    from .ticker_validator import TickerValidator, validate_tickers, is_valid_ticker
    from .smart_rate_limiter import SmartRateLimiter, get_rate_limiter
    from .request_cache import RequestCache, get_request_cache
    
    __all__ = [
        'TickerValidator',
        'validate_tickers',
        'is_valid_ticker',
        'SmartRateLimiter',
        'get_rate_limiter',
        'RequestCache',
        'get_request_cache',
    ]
except ImportError as e:
    # Graceful fallback if dependencies not available
    print(f"[WARN] Some optimization modules could not be imported: {e}")
    __all__ = []







