"""
Cache analytics and monitoring system.
Provides detailed insights into cache performance, hit rates, and optimization opportunities.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import threading
import statistics

logger = logging.getLogger(__name__)


class CacheMetrics:
    """Real-time cache metrics collection"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

        # Core metrics
        self.hits = deque(maxlen=window_size)
        self.misses = deque(maxlen=window_size)
        self.evictions = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)

        # Layer-specific metrics
        self.memory_hits = deque(maxlen=window_size)
        self.memory_misses = deque(maxlen=window_size)
        self.redis_hits = deque(maxlen=window_size)
        self.redis_misses = deque(maxlen=window_size)
        self.persistent_hits = deque(maxlen=window_size)
        self.persistent_misses = deque(maxlen=window_size)

        # Compression metrics
        self.compression_ratio = deque(maxlen=window_size)
        self.compression_savings = deque(maxlen=window_size)

        # Symbol-specific metrics
        self.symbol_access = defaultdict(lambda: {
            'hits': 0, 'misses': 0, 'response_times': deque(maxlen=100),
            'last_access': None
        })

        # Time tracking
        self.start_time = time.time()
        self.last_reset = time.time()

        self.lock = threading.Lock()

    def record_hit(self, symbol: str, response_time: float, layer: str = 'unknown'):
        """Record a cache hit"""
        with self.lock:
            self.hits.append(time.time())
            self.response_times.append(response_time)

            self.symbol_access[symbol]['hits'] += 1
            self.symbol_access[symbol]['response_times'].append(response_time)
            self.symbol_access[symbol]['last_access'] = time.time()

            # Layer-specific tracking
            if layer == 'memory':
                self.memory_hits.append(time.time())
            elif layer == 'redis':
                self.redis_hits.append(time.time())
            elif layer == 'persistent':
                self.persistent_hits.append(time.time())

    def record_miss(self, symbol: str, response_time: float, layer: str = 'unknown'):
        """Record a cache miss"""
        with self.lock:
            self.misses.append(time.time())
            self.response_times.append(response_time)

            self.symbol_access[symbol]['misses'] += 1
            self.symbol_access[symbol]['response_times'].append(response_time)
            self.symbol_access[symbol]['last_access'] = time.time()

            # Layer-specific tracking
            if layer == 'memory':
                self.memory_misses.append(time.time())
            elif layer == 'redis':
                self.redis_misses.append(time.time())
            elif layer == 'persistent':
                self.persistent_misses.append(time.time())

    def record_eviction(self, symbol: str):
        """Record a cache eviction"""
        with self.lock:
            self.evictions.append(time.time())

    def record_compression(self, original_size: int, compressed_size: int):
        """Record compression metrics"""
        with self.lock:
            if original_size > 0:
                ratio = compressed_size / original_size
                savings = original_size - compressed_size
                self.compression_ratio.append(ratio)
                self.compression_savings.append(savings)

    def get_hit_rate(self) -> float:
        """Calculate overall hit rate"""
        with self.lock:
            total_requests = len(self.hits) + len(self.misses)
            return (len(self.hits) / total_requests) if total_requests > 0 else 0.0

    def get_layer_hit_rates(self) -> Dict[str, float]:
        """Calculate hit rates for each cache layer"""
        with self.lock:
            return {
                'memory': self._calculate_hit_rate(self.memory_hits, self.memory_misses),
                'redis': self._calculate_hit_rate(self.redis_hits, self.redis_misses),
                'persistent': self._calculate_hit_rate(self.persistent_hits, self.persistent_misses)
            }

    def _calculate_hit_rate(self, hits: deque, misses: deque) -> float:
        """Calculate hit rate for specific layer"""
        total = len(hits) + len(misses)
        return (len(hits) / total) if total > 0 else 0.0

    def get_average_response_time(self) -> float:
        """Get average response time"""
        with self.lock:
            return statistics.mean(self.response_times) if self.response_times else 0.0

    def get_top_symbols(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most accessed symbols"""
        with self.lock:
            symbol_stats = []
            for symbol, data in self.symbol_access.items():
                total_requests = data['hits'] + data['misses']
                if total_requests > 0:
                    hit_rate = data['hits'] / total_requests
                    avg_response_time = (statistics.mean(data['response_times'])
                                       if data['response_times'] else 0.0)

                    symbol_stats.append({
                        'symbol': symbol,
                        'total_requests': total_requests,
                        'hits': data['hits'],
                        'misses': data['misses'],
                        'hit_rate': hit_rate,
                        'avg_response_time': avg_response_time,
                        'last_access': data['last_access']
                    })

            # Sort by total requests
            symbol_stats.sort(key=lambda x: x['total_requests'], reverse=True)
            return symbol_stats[:limit]

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        with self.lock:
            if not self.compression_ratio:
                return {'average_ratio': 0.0, 'total_savings_bytes': 0, 'samples': 0}

            return {
                'average_ratio': statistics.mean(self.compression_ratio),
                'total_savings_bytes': sum(self.compression_savings),
                'samples': len(self.compression_ratio)
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive cache performance summary"""
        with self.lock:
            uptime = time.time() - self.start_time
            total_requests = len(self.hits) + len(self.misses)

            return {
                'uptime_seconds': uptime,
                'total_requests': total_requests,
                'hits': len(self.hits),
                'misses': len(self.misses),
                'evictions': len(self.evictions),
                'hit_rate': self.get_hit_rate(),
                'average_response_time_ms': self.get_average_response_time() * 1000,
                'layer_hit_rates': self.get_layer_hit_rates(),
                'top_symbols': self.get_top_symbols(),
                'compression_stats': self.get_compression_stats(),
                'requests_per_second': total_requests / uptime if uptime > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }

    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.hits.clear()
            self.misses.clear()
            self.evictions.clear()
            self.response_times.clear()
            self.memory_hits.clear()
            self.memory_misses.clear()
            self.redis_hits.clear()
            self.redis_misses.clear()
            self.persistent_hits.clear()
            self.persistent_misses.clear()
            self.compression_ratio.clear()
            self.compression_savings.clear()
            self.symbol_access.clear()
            self.last_reset = time.time()


class CacheOptimizer:
    """Intelligent cache optimization recommendations"""

    def __init__(self, metrics: CacheMetrics):
        self.metrics = metrics

    def analyze_hot_symbols(self, threshold_requests: int = 100) -> List[str]:
        """Identify symbols that should be pre-warmed"""
        top_symbols = self.metrics.get_top_symbols(limit=50)
        hot_symbols = []

        for symbol_data in top_symbols:
            if symbol_data['total_requests'] >= threshold_requests:
                # Check if hit rate is good enough
                if symbol_data['hit_rate'] < 0.8:
                    hot_symbols.append(symbol_data['symbol'])

        return hot_symbols

    def recommend_ttl_adjustments(self) -> Dict[str, int]:
        """Recommend TTL adjustments based on access patterns"""
        recommendations = {}

        for symbol, data in self.metrics.symbol_access.items():
            total_requests = data['hits'] + data['misses']

            if total_requests < 10:  # Not enough data
                continue

            hit_rate = data['hits'] / total_requests

            # If hit rate is very high, consider longer TTL
            if hit_rate > 0.95:
                recommendations[symbol] = 7200  # 2 hours
            # If hit rate is moderate, keep current TTL
            elif hit_rate > 0.8:
                recommendations[symbol] = 1800  # 30 minutes
            # If hit rate is low, reduce TTL
            else:
                recommendations[symbol] = 300  # 5 minutes

        return recommendations

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive optimization recommendations"""
        hot_symbols = self.analyze_hot_symbols()
        ttl_recommendations = self.recommend_ttl_adjustments()

        # Analyze layer effectiveness
        layer_rates = self.metrics.get_layer_hit_rates()

        recommendations = []

        # Memory layer analysis
        if layer_rates['memory'] < 0.7:
            recommendations.append({
                'type': 'memory_optimization',
                'description': 'Memory cache hit rate is low. Consider increasing memory cache size or adjusting TTL settings.',
                'current_hit_rate': layer_rates['memory']
            })

        # Redis layer analysis
        if layer_rates['redis'] < 0.8:
            recommendations.append({
                'type': 'redis_optimization',
                'description': 'Redis cache hit rate could be improved. Consider pre-warming frequently accessed symbols.',
                'current_hit_rate': layer_rates['redis']
            })

        # Hot symbols analysis
        if hot_symbols:
            recommendations.append({
                'type': 'pre_warming',
                'description': f'Consider pre-warming these frequently accessed symbols: {hot_symbols[:5]}',
                'symbols': hot_symbols
            })

        # TTL optimization
        if ttl_recommendations:
            recommendations.append({
                'type': 'ttl_optimization',
                'description': 'TTL adjustments recommended for some symbols',
                'adjustments': ttl_recommendations
            })

        return {
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat(),
            'hot_symbols_count': len(hot_symbols),
            'ttl_recommendations_count': len(ttl_recommendations)
        }


class CacheMonitor:
    """Cache monitoring and alerting system"""

    def __init__(self, metrics: CacheMetrics, optimizer: CacheOptimizer):
        self.metrics = metrics
        self.optimizer = optimizer
        self.alerts = deque(maxlen=100)
        self.thresholds = {
            'min_hit_rate': 0.6,
            'max_response_time_ms': 2000,
            'max_eviction_rate': 0.1  # 10% eviction rate
        }

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for cache performance issues and generate alerts"""
        alerts = []

        # Hit rate alert
        hit_rate = self.metrics.get_hit_rate()
        if hit_rate < self.thresholds['min_hit_rate']:
            alerts.append({
                'level': 'WARNING',
                'type': 'low_hit_rate',
                'message': f'Cache hit rate is low: {hit_rate:.2%}',
                'current_value': hit_rate,
                'threshold': self.thresholds['min_hit_rate'],
                'timestamp': datetime.now().isoformat()
            })

        # Response time alert
        avg_response_time = self.metrics.get_average_response_time() * 1000
        if avg_response_time > self.thresholds['max_response_time_ms']:
            alerts.append({
                'level': 'ERROR',
                'type': 'high_response_time',
                'message': f'Average response time is high: {avg_response_time:.0f}ms',
                'current_value': avg_response_time,
                'threshold': self.thresholds['max_response_time_ms'],
                'timestamp': datetime.now().isoformat()
            })

        # Eviction rate alert
        total_requests = len(self.metrics.hits) + len(self.metrics.misses)
        if total_requests > 0:
            eviction_rate = len(self.metrics.evictions) / total_requests
            if eviction_rate > self.thresholds['max_eviction_rate']:
                alerts.append({
                    'level': 'WARNING',
                    'type': 'high_eviction_rate',
                    'message': f'Eviction rate is high: {eviction_rate:.2%}',
                    'current_value': eviction_rate,
                    'threshold': self.thresholds['max_eviction_rate'],
                    'timestamp': datetime.now().isoformat()
                })

        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)

        return alerts

    def get_alert_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return list(self.alerts)[-limit:]

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get complete monitoring dashboard"""
        summary = self.metrics.get_summary()
        recommendations = self.optimizer.get_optimization_recommendations()
        alerts = self.check_alerts()

        return {
            'summary': summary,
            'recommendations': recommendations,
            'current_alerts': alerts,
            'alert_history': self.get_alert_history(),
            'thresholds': self.thresholds
        }


# Global instances
_metrics = None
_optimizer = None
_monitor = None

def get_cache_analytics() -> Tuple[CacheMetrics, CacheOptimizer, CacheMonitor]:
    """Get or create global cache analytics instances"""
    global _metrics, _optimizer, _monitor

    if _metrics is None:
        _metrics = CacheMetrics()

    if _optimizer is None:
        _optimizer = CacheOptimizer(_metrics)

    if _monitor is None:
        _monitor = CacheMonitor(_metrics, _optimizer)

    return _metrics, _optimizer, _monitor

def record_cache_hit(symbol: str, response_time: float, layer: str = 'unknown'):
    """Convenience function to record cache hit"""
    metrics, _, _ = get_cache_analytics()
    metrics.record_hit(symbol, response_time, layer)

def record_cache_miss(symbol: str, response_time: float, layer: str = 'unknown'):
    """Convenience function to record cache miss"""
    metrics, _, _ = get_cache_analytics()
    metrics.record_miss(symbol, response_time, layer)

def get_cache_performance_report() -> Dict[str, Any]:
    """Get complete cache performance report"""
    _, _, monitor = get_cache_analytics()
    return monitor.get_monitoring_dashboard()