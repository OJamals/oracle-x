"""
Cache warming service for preloading popular ticker data at market open.
Provides intelligent cache warming based on trading volume, market cap, and access patterns.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

try:
    import schedule  # type: ignore
except Exception:
    schedule = None  # Schedule is optional; warming will be disabled if missing

logger = logging.getLogger(__name__)


@dataclass
class MarketHours:
    """Market hours configuration"""

    pre_market_start: str = "04:00"  # ET
    market_open: str = "09:30"  # ET
    market_close: str = "16:00"  # ET
    after_hours_end: str = "20:00"  # ET

    def is_market_hours(self) -> bool:
        """Check if current time is during market hours"""
        now = datetime.now(self.get_eastern_timezone())
        current_time = now.strftime("%H:%M")

        # Weekend check
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        return self.pre_market_start <= current_time <= self.after_hours_end

    def is_open_hours(self) -> bool:
        """Check if current time is during regular trading hours"""
        now = datetime.now(self.get_eastern_timezone())
        current_time = now.strftime("%H:%M")

        # Weekend check
        if now.weekday() >= 5:
            return False

        return self.market_open <= current_time <= self.market_close

    def get_eastern_timezone(self):
        """Get Eastern timezone (simplified - could use pytz)"""
        # For simplicity, assuming system is in ET or converting
        return None  # Will use local time


@dataclass
class CacheWarmupConfig:
    """Configuration for cache warming"""

    enabled: bool = True
    warm_up_at_market_open: bool = True
    warm_up_hourly: bool = True
    warm_up_interval_minutes: int = 60
    popular_symbols: List[str] = field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
            "BABA",
            "ORCL",
            "CRM",
            "AMD",
            "INTC",
            "UBER",
            "SPY",
            "QQQ",
        ]
    )
    mid_cap_symbols: List[str] = field(
        default_factory=lambda: [
            "INTC",
            "AMD",
            "UBER",
            "LYFT",
            "SQ",
            "SHOP",
            "CRWD",
            "DDOG",
            "ZS",
            "NET",
            "OKTA",
            "MDB",
            "TEAM",
            "PANW",
            "FTNT",
        ]
    )
    volume_threshold: int = 1000000  # Minimum volume for cache warming

    @classmethod
    def from_env(cls) -> "CacheWarmupConfig":
        """Create config from environment variables"""
        return cls(
            enabled=os.getenv("CACHE_WARMING_ENABLED", "true").lower() == "true",
            warm_up_at_market_open=os.getenv("WARM_UP_AT_MARKET_OPEN", "true").lower()
            == "true",
            warm_up_hourly=os.getenv("WARM_UP_HOURLY", "true").lower() == "true",
            warm_up_interval_minutes=int(os.getenv("WARM_UP_INTERVAL_MINUTES", "60")),
            volume_threshold=int(
                os.getenv("CACHE_WARMING_VOLUME_THRESHOLD", "1000000")
            ),
        )


class CacheWarmingService:
    """Service for intelligent cache warming"""

    def __init__(self, orchestrator, config: Optional[CacheWarmupConfig] = None):
        self.orchestrator = orchestrator
        self.config = config or CacheWarmupConfig.from_env()
        self.market_hours = MarketHours()
        self.is_running = False
        self.warming_thread = None
        self.access_patterns = {}  # Track frequently accessed symbols
        self.last_warmup_time = None

        # Scheduler for automated warming (disabled if schedule is unavailable)
        self.scheduler = schedule.Scheduler() if schedule else None

    def start(self):
        """Start the cache warming service"""
        if not self.scheduler:
            logger.warning(
                "Cache warming service not started (schedule dependency missing)"
            )
            return
        if self.is_running:
            logger.warning("Cache warming service is already running")
            return

        self.is_running = True
        logger.info("Starting cache warming service")

        # Schedule market open warming
        if self.config.warm_up_at_market_open:
            self.scheduler.every().day.at(self.market_hours.market_open).do(
                self._warm_up_at_market_open
            )

        # Schedule hourly warming during market hours
        if self.config.warm_up_hourly:
            self.scheduler.every(self.config.warm_up_interval_minutes).minutes.do(
                self._periodic_warm_up
            )

        # Start background thread
        self.warming_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.warming_thread.start()

        logger.info("Cache warming service started successfully")

    def stop(self):
        """Stop the cache warming service"""
        if not self.is_running or not self.scheduler:
            return

        self.is_running = False
        self.scheduler.clear()
        logger.info("Cache warming service stopped")

    def _run_scheduler(self):
        """Run the scheduler in background thread"""
        while self.is_running and self.scheduler:
            try:
                self.scheduler.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in cache warming scheduler: {e}")
                time.sleep(60)  # Wait longer on error

    def _warm_up_at_market_open(self):
        """Warm up cache at market open with comprehensive data"""
        if not self.config.enabled:
            return

        logger.info("Starting market open cache warm-up")

        try:
            # Warm up popular symbols with full data
            all_symbols = self.config.popular_symbols + self.config.mid_cap_symbols

            # First, get quotes for all symbols (fastest)
            for symbol in all_symbols:
                try:
                    quote = self.orchestrator.get_enhanced_quote(symbol)
                    if (
                        quote
                        and quote.volume
                        and quote.volume > self.config.volume_threshold
                    ):
                        logger.debug(f"Market open warm-up: Quote cached for {symbol}")
                except Exception as e:
                    logger.debug(f"Failed to warm up quote for {symbol}: {e}")

            # Then warm up market data for popular symbols only
            for symbol in self.config.popular_symbols:
                try:
                    market_data = self.orchestrator.get_enhanced_market_data(
                        symbol, period="3mo", interval="1d"
                    )
                    if market_data is not None:
                        logger.debug(
                            f"Market open warm-up: Market data cached for {symbol}"
                        )
                except Exception as e:
                    logger.debug(f"Failed to warm up market data for {symbol}: {e}")

            # Finally, warm up sentiment for top symbols
            for symbol in self.config.popular_symbols[:8]:  # Top 8 only
                try:
                    sentiment = self.orchestrator.get_enhanced_sentiment_data(symbol)
                    if sentiment:
                        logger.debug(
                            f"Market open warm-up: Sentiment cached for {symbol}"
                        )
                except Exception as e:
                    logger.debug(f"Failed to warm up sentiment for {symbol}: {e}")

            self.last_warmup_time = datetime.now()
            logger.info(
                f"Market open cache warm-up completed for {len(all_symbols)} symbols"
            )

        except Exception as e:
            logger.error(f"Error during market open cache warm-up: {e}")

    def _periodic_warm_up(self):
        """Periodic cache warm-up during market hours"""
        if not self.config.enabled or not self.market_hours.is_market_hours():
            return

        try:
            # Warm up popular symbols with lightweight data
            popular_symbols = self.config.popular_symbols

            for symbol in popular_symbols:
                try:
                    # Only warm up quotes during periodic warming (lighter)
                    quote = self.orchestrator.get_enhanced_quote(symbol)
                    if quote:
                        logger.debug(f"Periodic warm-up: Quote cached for {symbol}")
                except Exception as e:
                    logger.debug(f"Failed to warm up quote for {symbol}: {e}")

            self.last_warmup_time = datetime.now()
            logger.debug(
                f"Periodic cache warm-up completed for {len(popular_symbols)} symbols"
            )

        except Exception as e:
            logger.error(f"Error during periodic cache warm-up: {e}")

    def track_access(self, symbol: str):
        """Track symbol access patterns for intelligent warming"""
        if symbol not in self.access_patterns:
            self.access_patterns[symbol] = {
                "access_count": 0,
                "last_access": None,
                "first_access": datetime.now(),
            }

        self.access_patterns[symbol]["access_count"] += 1
        self.access_patterns[symbol]["last_access"] = datetime.now()

    def get_popular_symbols(self, limit: int = 20) -> List[str]:
        """Get most popular symbols based on access patterns"""
        # Sort by access count and recency
        now = datetime.now()
        scored_symbols = []

        for symbol, data in self.access_patterns.items():
            # Score based on access count and recency
            hours_since_access = (now - data["last_access"]).total_seconds() / 3600
            recency_score = max(0, 1 - (hours_since_access / 24))  # Decay over 24 hours
            score = data["access_count"] * recency_score
            scored_symbols.append((symbol, score))

        # Sort by score and return top symbols
        scored_symbols.sort(key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in scored_symbols[:limit]]

    def warm_up_based_on_access_patterns(self, limit: int = 10):
        """Warm up cache based on access patterns"""
        if not self.config.enabled:
            return

        popular_from_access = self.get_popular_symbols(limit)
        logger.info(
            f"Warming up cache for {len(popular_from_access)} frequently accessed symbols"
        )

        for symbol in popular_from_access:
            try:
                quote = self.orchestrator.get_enhanced_quote(symbol)
                if quote:
                    logger.debug(f"Access-based warm-up: Quote cached for {symbol}")
            except Exception as e:
                logger.debug(f"Failed to warm up quote for {symbol}: {e}")

    def get_service_status(self) -> Dict[str, Any]:
        """Get cache warming service status"""
        return {
            "is_running": self.is_running,
            "config": {
                "enabled": self.config.enabled,
                "warm_up_at_market_open": self.config.warm_up_at_market_open,
                "warm_up_hourly": self.config.warm_up_hourly,
                "interval_minutes": self.config.warm_up_interval_minutes,
                "popular_symbols_count": len(self.config.popular_symbols),
                "volume_threshold": self.config.volume_threshold,
            },
            "market_status": {
                "is_market_hours": self.market_hours.is_market_hours(),
                "is_open_hours": self.market_hours.is_open_hours(),
                "current_time": datetime.now().strftime("%H:%M:%S %Z"),
            },
            "access_patterns": {
                "tracked_symbols": len(self.access_patterns),
                "most_accessed": self.get_popular_symbols(5),
            },
            "last_warmup_time": (
                self.last_warmup_time.isoformat() if self.last_warmup_time else None
            ),
        }


def create_cache_warming_service(orchestrator) -> CacheWarmingService:
    """Create and configure cache warming service"""
    if not schedule:
        logger.info("Cache warming disabled: schedule dependency not installed")
        return None

    config = CacheWarmupConfig.from_env()
    service = CacheWarmingService(orchestrator, config)

    if config.enabled:
        service.start()

    return service


# Integration functions for easy access
_warming_service = None


def get_cache_warming_service(orchestrator=None):
    """Get or create global cache warming service instance"""
    global _warming_service

    if _warming_service is None and orchestrator:
        _warming_service = create_cache_warming_service(orchestrator)

    return _warming_service


def start_cache_warming(orchestrator):
    """Start cache warming service"""
    service = get_cache_warming_service(orchestrator)
    if service and service.scheduler and not service.is_running:
        service.start()


def stop_cache_warming():
    """Stop cache warming service"""
    global _warming_service
    if _warming_service and _warming_service.is_running:
        _warming_service.stop()
        _warming_service = None
