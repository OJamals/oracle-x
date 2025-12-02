"""
Oracle-X Async Data Fetching Module (Phase 2 Optimization)

This module provides async versions of all data fetching functions,
enabling parallel execution for dramatic performance improvement.

Expected improvement: 81s → 25-30s (60-65% faster)
"""

import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import functools

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=4)


def async_wrap(func):
    """
    Decorator to convert synchronous function to async.
    Runs sync function in thread pool to avoid blocking event loop.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))
    return wrapper


# ============================================================================
# Async Market Internals
# ============================================================================

@async_wrap
def fetch_market_internals_sync():
    """Sync version wrapped for async execution"""
    from data_feeds.market_internals import fetch_market_internals
    return fetch_market_internals()


async def fetch_market_internals_async() -> Dict[str, Any]:
    """
    Async version of fetch_market_internals.
    Fetches market internals data without blocking.
    
    Returns:
        dict: Market internals snapshot
    """
    try:
        logger.debug("[ASYNC] Fetching market internals...")
        start_time = asyncio.get_event_loop().time()
        
        result = await fetch_market_internals_sync()
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"[ASYNC] Market internals fetched in {elapsed:.2f}s")
        
        return result
    except Exception as e:
        logger.error(f"[ASYNC] Error fetching market internals: {e}")
        return {}


# ============================================================================
# Async Options Flow
# ============================================================================

@async_wrap
def fetch_options_flow_sync(tickers: List[str]):
    """Sync version wrapped for async execution"""
    from data_feeds.options_flow import fetch_options_flow
    return fetch_options_flow(tickers)


async def fetch_options_flow_async(tickers: List[str]) -> Dict[str, Any]:
    """
    Async version of fetch_options_flow.
    Fetches options flow data without blocking.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        dict: Options flow data with unusual sweeps
    """
    try:
        logger.debug(f"[ASYNC] Fetching options flow for {len(tickers)} tickers...")
        start_time = asyncio.get_event_loop().time()
        
        result = await fetch_options_flow_sync(tickers)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"[ASYNC] Options flow fetched in {elapsed:.2f}s")
        
        return result
    except Exception as e:
        logger.error(f"[ASYNC] Error fetching options flow: {e}")
        return {"unusual_sweeps": [], "total_sweeps": 0}


# ============================================================================
# Async Dark Pool Data
# ============================================================================

@async_wrap
def fetch_dark_pool_data_sync(tickers: List[str]):
    """Sync version wrapped for async execution"""
    from data_feeds.dark_pools import fetch_dark_pool_data
    return fetch_dark_pool_data(tickers)


async def fetch_dark_pool_data_async(tickers: List[str]) -> Dict[str, Any]:
    """
    Async version of fetch_dark_pool_data.
    Fetches dark pool signals without blocking.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        dict: Dark pool activity data
    """
    try:
        logger.debug(f"[ASYNC] Fetching dark pool data for {len(tickers)} tickers...")
        start_time = asyncio.get_event_loop().time()
        
        result = await fetch_dark_pool_data_sync(tickers)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"[ASYNC] Dark pool data fetched in {elapsed:.2f}s")
        
        return result
    except Exception as e:
        logger.error(f"[ASYNC] Error fetching dark pool data: {e}")
        return {"dark_pools": [], "total_detected": 0}


# ============================================================================
# Async Sentiment Analysis
# ============================================================================

@async_wrap
def fetch_sentiment_data_sync(tickers: List[str]):
    """Sync version wrapped for async execution"""
    from data_feeds.sentiment import fetch_sentiment_data
    return fetch_sentiment_data(tickers)


async def fetch_sentiment_data_async(tickers: List[str]) -> Dict[str, Any]:
    """
    Async version of fetch_sentiment_data.
    Fetches sentiment data from Reddit and Twitter without blocking.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        dict: Sentiment scores per ticker
    """
    try:
        logger.debug(f"[ASYNC] Fetching sentiment data for {len(tickers)} tickers...")
        start_time = asyncio.get_event_loop().time()
        
        result = await fetch_sentiment_data_sync(tickers)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"[ASYNC] Sentiment data fetched in {elapsed:.2f}s")
        
        return result
    except Exception as e:
        logger.error(f"[ASYNC] Error fetching sentiment data: {e}")
        return {}


# ============================================================================
# Async News Headlines
# ============================================================================

@async_wrap
def fetch_headlines_yahoo_sync():
    """Sync version wrapped for async execution"""
    from data_feeds.news_scraper import fetch_headlines_yahoo_finance
    return fetch_headlines_yahoo_finance()


async def fetch_headlines_yahoo_async() -> List[str]:
    """
    Async version of fetch_headlines_yahoo_finance.
    Fetches Yahoo Finance headlines without blocking.
    
    Returns:
        list: News headlines
    """
    try:
        logger.debug("[ASYNC] Fetching Yahoo headlines...")
        start_time = asyncio.get_event_loop().time()
        
        result = await fetch_headlines_yahoo_sync()
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"[ASYNC] Yahoo headlines fetched in {elapsed:.2f}s")
        
        return result
    except Exception as e:
        logger.error(f"[ASYNC] Error fetching Yahoo headlines: {e}")
        return []


# ============================================================================
# Async Finviz Breadth
# ============================================================================

@async_wrap
def fetch_finviz_breadth_sync():
    """Sync version wrapped for async execution"""
    from data_feeds.finviz_scraper import fetch_finviz_breadth
    return fetch_finviz_breadth()


async def fetch_finviz_breadth_async() -> Dict[str, Any]:
    """
    Async version of fetch_finviz_breadth.
    Fetches market breadth data without blocking.
    
    Returns:
        dict: Market breadth statistics
    """
    try:
        logger.debug("[ASYNC] Fetching Finviz breadth...")
        start_time = asyncio.get_event_loop().time()
        
        result = await fetch_finviz_breadth_sync()
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"[ASYNC] Finviz breadth fetched in {elapsed:.2f}s")
        
        return result
    except Exception as e:
        logger.error(f"[ASYNC] Error fetching Finviz breadth: {e}")
        return {}


# ============================================================================
# Async Earnings Calendar
# ============================================================================

@async_wrap
def fetch_earnings_calendar_sync(tickers: List[str]):
    """Sync version wrapped for async execution"""
    from data_feeds.earnings_calendar import fetch_earnings_calendar
    return fetch_earnings_calendar(tickers)


async def fetch_earnings_calendar_async(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Async version of fetch_earnings_calendar.
    Fetches earnings dates without blocking.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        list: Earnings calendar entries
    """
    try:
        logger.debug(f"[ASYNC] Fetching earnings calendar for {len(tickers)} tickers...")
        start_time = asyncio.get_event_loop().time()
        
        result = await fetch_earnings_calendar_sync(tickers)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"[ASYNC] Earnings calendar fetched in {elapsed:.2f}s")
        
        return result
    except Exception as e:
        logger.error(f"[ASYNC] Error fetching earnings calendar: {e}")
        return []


# ============================================================================
# Async Signal Orchestrator (Main Entry Point)
# ============================================================================

async def get_signals_from_scrapers_async(
    prompt_text: str,
    chart_image_b64: Optional[str] = None,
    enable_premium: bool = True
) -> Dict[str, Any]:
    """
    Async version of get_signals_from_scrapers.
    Fetches ALL data sources concurrently for maximum performance.
    
    This is the main entry point for Phase 2 optimization.
    Expected improvement: 40-50 seconds → 15-20 seconds
    
    Args:
        prompt_text: User prompt or market summary
        chart_image_b64: Base64-encoded chart image (optional)
        enable_premium: Enable premium API calls (default: True)
        
    Returns:
        dict: All signals snapshot from parallel data fetching
    """
    logger.info("🚀 [PHASE 2] Starting async signal collection...")
    start_time = asyncio.get_event_loop().time()
    
    # Get ticker universe (with validation from Phase 1)
    from data_feeds.ticker_universe import fetch_ticker_universe
    tickers = fetch_ticker_universe(sample_size=40)
    logger.info(f"📊 Analyzing {len(tickers)} tickers")
    
    # PHASE 2 OPTIMIZATION: Launch ALL tasks concurrently
    tasks = {
        'market_internals': fetch_market_internals_async(),
        'options_flow': fetch_options_flow_async(tickers),
        'dark_pools': fetch_dark_pool_data_async(tickers),
        'sentiment_web': fetch_sentiment_data_async(tickers),
        'yahoo_headlines': fetch_headlines_yahoo_async(),
        'finviz_breadth': fetch_finviz_breadth_async(),
        'earnings_calendar': fetch_earnings_calendar_async(tickers),
    }
    
    # Wait for ALL tasks to complete concurrently
    logger.info(f"⚡ Launching {len(tasks)} concurrent tasks...")
    
    try:
        # Gather with timeout to prevent hanging
        results = await asyncio.wait_for(
            asyncio.gather(*tasks.values(), return_exceptions=True),
            timeout=60.0  # 60 second timeout
        )
    except asyncio.TimeoutError:
        logger.error("⚠️  Some data fetching timed out after 60s")
        results = [None] * len(tasks)
    
    # Unpack results
    task_names = list(tasks.keys())
    signals = {}
    
    for i, (name, result) in enumerate(zip(task_names, results)):
        if isinstance(result, Exception):
            logger.warning(f"⚠️  {name} failed: {result}")
            signals[name] = {} if name != 'yahoo_headlines' else []
        else:
            signals[name] = result
    
    # Add tickers to signals
    signals['tickers'] = tickers
    
    # Add synchronous components (fast, no benefit from async)
    from oracle_engine.prompt_chain import get_sentiment, analyze_chart
    
    logger.debug("Adding LLM sentiment analysis...")
    signals['sentiment_llm'] = get_sentiment(prompt_text)
    
    if chart_image_b64:
        logger.debug("Adding chart analysis...")
        signals['chart_analysis'] = analyze_chart(chart_image_b64)
    else:
        signals['chart_analysis'] = "No chart provided."
    
    # Handle synthetic dark pool signals
    try:
        from data_feeds.synthetic_darkpool_signals import generate_synthetic_darkpool_signals
        
        dark_pools_enhanced = generate_synthetic_darkpool_signals(
            options_data=signals.get('options_flow', {}),
            real_darkpool_data=signals.get('dark_pools', {}).get('dark_pools', [])
        )
        
        if dark_pools_enhanced:
            signals['dark_pools'] = {
                "dark_pools": dark_pools_enhanced,
                "data_source": "synthetic_options_correlation",
                "timestamp": datetime.now().isoformat(),
                "total_detected": len(dark_pools_enhanced)
            }
            logger.info(f"✅ Enhanced dark pool detection: {len(dark_pools_enhanced)} signals")
    except Exception as e:
        logger.warning(f"⚠️  Synthetic dark pool generation failed: {e}")
    
    # Add premium data if enabled
    if enable_premium:
        try:
            logger.debug("Fetching premium data...")
            from data_feeds.strategic_premium_feeds import fetch_premium_unique_data
            premium_data = await asyncio.to_thread(
                fetch_premium_unique_data,
                tickers[:5]  # Limit to top 5 for premium
            )
            signals['premium_data'] = premium_data
            logger.info("✅ Premium data fetched")
        except Exception as e:
            logger.warning(f"⚠️  Premium data fetch failed: {e}")
            signals['premium_data'] = {}
    
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(f"✅ [PHASE 2] Async signal collection completed in {elapsed:.2f}s")
    logger.info(f"   Speedup vs sequential: ~{elapsed/40:.1f}x faster (estimated)")
    
    return signals


# ============================================================================
# Utility Functions
# ============================================================================

def cleanup_async_resources():
    """Clean up async resources (call on shutdown)"""
    _executor.shutdown(wait=False)
    logger.info("Async resources cleaned up")


# Example usage
if __name__ == "__main__":
    async def test_async_pipeline():
        """Test the async pipeline"""
        print("Testing async signal collection...")
        
        signals = await get_signals_from_scrapers_async(
            "Test market analysis",
            chart_image_b64=None,
            enable_premium=False
        )
        
        print(f"\nResults:")
        print(f"  Tickers: {len(signals.get('tickers', []))}")
        print(f"  Options sweeps: {signals.get('options_flow', {}).get('total_sweeps', 0)}")
        print(f"  Dark pools: {signals.get('dark_pools', {}).get('total_detected', 0)}")
        print(f"  Headlines: {len(signals.get('yahoo_headlines', []))}")
    
    # Run test
    asyncio.run(test_async_pipeline())







