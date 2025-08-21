"""
Cache invalidation service for managing stale data and ensuring cache freshness.
Provides intelligent invalidation strategies based on data age, quality, and market conditions.
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import schedule

logger = logging.getLogger(__name__)


@dataclass
class InvalidationRule:
    """Rule for cache invalidation"""
    name: str
    data_type: str
    max_age_seconds: int
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, higher = more important


@dataclass
class InvalidationStats:
    """Statistics for cache invalidation operations"""
    total_invalidated: int = 0
    rules_executed: int = 0
    last_run_time: Optional[datetime] = None
    errors: int = 0
    by_data_type: Dict[str, int] = field(default_factory=dict)


class CacheInvalidationService:
    """Intelligent cache invalidation service"""

    def __init__(self, redis_cache_manager, orchestrator=None):
        self.redis_cache = redis_cache_manager
        self.orchestrator = orchestrator
        self.is_running = False
        self.invalidation_thread = None
        self.stats = InvalidationStats()

        # Default invalidation rules
        self.rules = [
            InvalidationRule(
                name="stale_quotes",
                data_type="ticker_data",
                max_age_seconds=300,  # 5 minutes
                priority=8
            ),
            InvalidationRule(
                name="old_market_data",
                data_type="market_data",
                max_age_seconds=1800,  # 30 minutes
                priority=6
            ),
            InvalidationRule(
                name="outdated_sentiment",
                data_type="sentiment",
                max_age_seconds=3600,  # 1 hour
                priority=5
            ),
            InvalidationRule(
                name="news_cleanup",
                data_type="news",
                max_age_seconds=7200,  # 2 hours
                priority=4
            )
        ]

        # Scheduler for automated invalidation
        self.scheduler = schedule.Scheduler()

    def start(self):
        """Start the cache invalidation service"""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Starting cache invalidation service")

        # Schedule periodic cleanup
        self.scheduler.every(10).minutes.do(self._run_invalidations)
        # Schedule hourly deep cleanup
        self.scheduler.every().hour.do(self._deep_cleanup)

        # Start background thread
        self.invalidation_thread = threading.Thread(
            target=self._run_scheduler, daemon=True
        )
        self.invalidation_thread.start()

        logger.info("Cache invalidation service started")

    def stop(self):
        """Stop the cache invalidation service"""
        if not self.is_running:
            return

        self.is_running = False
        self.scheduler.clear()
        logger.info("Cache invalidation service stopped")

    def _run_scheduler(self):
        """Run the scheduler in background thread"""
        while self.is_running:
            try:
                self.scheduler.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in invalidation scheduler: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def add_rule(self, rule: InvalidationRule):
        """Add a new invalidation rule"""
        self.rules.append(rule)
        # Sort by priority (higher first)
        self.rules.sort(key=lambda x: x.priority, reverse=True)
        logger.info(f"Added invalidation rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove an invalidation rule"""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                logger.info(f"Removed invalidation rule: {rule_name}")
                return True
        return False

    def _run_invalidations(self):
        """Run all invalidation rules"""
        if not self.redis_cache:
            return

        logger.debug("Running cache invalidation rules")
        total_invalidated = 0

        # Sort rules by priority
        sorted_rules = sorted(self.rules, key=lambda x: x.priority, reverse=True)

        for rule in sorted_rules:
            try:
                invalidated = self._apply_rule(rule)
                total_invalidated += invalidated
                self.stats.rules_executed += 1

                if invalidated > 0:
                    self.stats.by_data_type[rule.data_type] = (
                        self.stats.by_data_type.get(rule.data_type, 0) + invalidated
                    )
                    logger.info(f"Rule '{rule.name}' invalidated {invalidated} items")

            except Exception as e:
                logger.error(f"Error applying rule '{rule.name}': {e}")
                self.stats.errors += 1

        self.stats.total_invalidated += total_invalidated
        self.stats.last_run_time = datetime.now()

        if total_invalidated > 0:
            logger.info(f"Cache invalidation completed: {total_invalidated} items invalidated")

    def _apply_rule(self, rule: InvalidationRule) -> int:
        """Apply a single invalidation rule"""
        invalidated = 0

        try:
            # Get all keys for this data type
            pattern = f"*{rule.data_type}*"
            keys = self.redis_cache._redis.keys(pattern) if self.redis_cache._redis else []

            for key in keys:
                try:
                    # Check if key matches rule conditions
                    if self._should_invalidate_key(key, rule):
                        # Invalidate the key
                        self.redis_cache._redis.delete(key)
                        invalidated += 1

                except Exception as e:
                    logger.debug(f"Error checking key {key}: {e}")

        except Exception as e:
            logger.error(f"Error applying rule {rule.name}: {e}")

        return invalidated

    def _should_invalidate_key(self, key: bytes, rule: InvalidationRule) -> bool:
        """Check if a key should be invalidated based on rule"""
        try:
            key_str = key.decode('utf-8')

            # Check age if Redis supports it
            if self.redis_cache._redis:
                # Get TTL of the key
                ttl = self.redis_cache._redis.ttl(key)
                if ttl == -1:  # Key has no expiration
                    # Check if it should be invalidated based on age
                    # This is a simplified check - in production you'd want to store creation time
                    return True

                # If TTL is greater than max_age, it might be too long
                if ttl > rule.max_age_seconds:
                    return True

        except Exception as e:
            logger.debug(f"Error checking invalidation for key {key}: {e}")

        return False

    def _deep_cleanup(self):
        """Perform deep cleanup of stale data"""
        logger.info("Running deep cache cleanup")

        try:
            if self.redis_cache and self.redis_cache._redis:
                # Get all keys
                keys = self.redis_cache._redis.keys('*')

                for key in keys:
                    try:
                        # Check if key is very old (no TTL or very long TTL)
                        ttl = self.redis_cache._redis.ttl(key)

                        # If key has no expiration or TTL > 24 hours, consider it for cleanup
                        if ttl == -1 or ttl > 86400:  # 24 hours
                            # Additional check: if key contains old data patterns
                            key_str = key.decode('utf-8')
                            if any(old_pattern in key_str.lower() for old_pattern in
                                   ['old', 'stale', 'expired', 'temp']):
                                self.redis_cache._redis.delete(key)
                                logger.debug(f"Deep cleanup: removed key {key_str}")

                    except Exception as e:
                        logger.debug(f"Error in deep cleanup for key {key}: {e}")

        except Exception as e:
            logger.error(f"Error in deep cleanup: {e}")

    def invalidate_by_symbol(self, symbol: str) -> int:
        """Invalidate all cache entries for a specific symbol"""
        if not self.redis_cache or not self.redis_cache._redis:
            return 0

        invalidated = 0
        try:
            # Find all keys containing the symbol
            pattern = f"*{symbol.upper()}*"
            keys = self.redis_cache._redis.keys(pattern)

            for key in keys:
                try:
                    self.redis_cache._redis.delete(key)
                    invalidated += 1
                except Exception as e:
                    logger.debug(f"Error deleting key {key}: {e}")

        except Exception as e:
            logger.error(f"Error invalidating symbol {symbol}: {e}")

        if invalidated > 0:
            logger.info(f"Invalidated {invalidated} cache entries for symbol {symbol}")

        return invalidated

    def invalidate_by_data_type(self, data_type: str) -> int:
        """Invalidate all cache entries for a specific data type"""
        if not self.redis_cache or not self.redis_cache._redis:
            return 0

        invalidated = 0
        try:
            pattern = f"*{data_type}*"
            keys = self.redis_cache._redis.keys(pattern)

            for key in keys:
                try:
                    self.redis_cache._redis.delete(key)
                    invalidated += 1
                except Exception as e:
                    logger.debug(f"Error deleting key {key}: {e}")

        except Exception as e:
            logger.error(f"Error invalidating data type {data_type}: {e}")

        if invalidated > 0:
            logger.info(f"Invalidated {invalidated} cache entries for data type {data_type}")

        return invalidated

    def force_cleanup(self, max_age_seconds: int = 3600) -> int:
        """Force cleanup of entries older than specified age"""
        if not self.redis_cache or not self.redis_cache._redis:
            return 0

        invalidated = 0
        try:
            keys = self.redis_cache._redis.keys('*')

            for key in keys:
                try:
                    ttl = self.redis_cache._redis.ttl(key)
                    if ttl == -1 or ttl > max_age_seconds:
                        self.redis_cache._redis.delete(key)
                        invalidated += 1
                except Exception as e:
                    logger.debug(f"Error checking key {key}: {e}")

        except Exception as e:
            logger.error(f"Error in force cleanup: {e}")

        if invalidated > 0:
            logger.info(f"Force cleanup completed: {invalidated} entries removed")

        return invalidated

    def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get invalidation service statistics"""
        return {
            'is_running': self.is_running,
            'stats': {
                'total_invalidated': self.stats.total_invalidated,
                'rules_executed': self.stats.rules_executed,
                'errors': self.stats.errors,
                'last_run_time': self.stats.last_run_time.isoformat() if self.stats.last_run_time else None,
                'by_data_type': dict(self.stats.by_data_type)
            },
            'rules': [
                {
                    'name': rule.name,
                    'data_type': rule.data_type,
                    'max_age_seconds': rule.max_age_seconds,
                    'priority': rule.priority
                }
                for rule in self.rules
            ],
            'timestamp': datetime.now().isoformat()
        }


# Global instance
_invalidation_service = None

def get_cache_invalidation_service(redis_cache_manager, orchestrator=None):
    """Get or create global cache invalidation service instance"""
    global _invalidation_service

    if _invalidation_service is None:
        _invalidation_service = CacheInvalidationService(
            redis_cache_manager, orchestrator
        )

    return _invalidation_service

def start_cache_invalidation(redis_cache_manager, orchestrator=None):
    """Start cache invalidation service"""
    service = get_cache_invalidation_service(redis_cache_manager, orchestrator)
    if not service.is_running:
        service.start()

def stop_cache_invalidation():
    """Stop cache invalidation service"""
    global _invalidation_service
    if _invalidation_service and _invalidation_service.is_running:
        _invalidation_service.stop()