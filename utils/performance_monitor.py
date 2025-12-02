"""
Performance monitoring system for Oracle-X financial trading system.
Provides comprehensive performance tracking, metrics collection, and optimization insights.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from functools import wraps
import statistics
import json

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Create dummy classes for when prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

        def inc(self, amount=1):
            pass

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

        def set(self, value):
            pass

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

        def observe(self, value):
            pass

    def start_http_server(*args, **kwargs):
        pass


from core.cache.redis_manager import get_cache_manager
from core.cache.sqlite_manager import get_sqlite_cache_manager

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""

    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()


class PerformanceLevel(Enum):
    """Performance level classifications."""

    OPTIMAL = auto()
    ACCEPTABLE = auto()
    DEGRADED = auto()
    CRITICAL = auto()


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""

    warning: float
    critical: float
    unit: str = "ms"


@dataclass
class ComponentMetrics:
    """Performance metrics for a specific component."""

    component: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    last_call_time: Optional[datetime] = None
    thresholds: Dict[str, PerformanceThreshold] = field(default_factory=dict)

    @property
    def average_time(self) -> float:
        """Calculate average call time."""
        if self.total_calls == 0:
            return 0.0
        return self.total_time / self.total_calls

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def current_level(self) -> PerformanceLevel:
        """Get current performance level."""
        avg_time = self.average_time
        if avg_time == 0:
            return PerformanceLevel.OPTIMAL

        # Check against thresholds if configured
        if "response_time" in self.thresholds:
            threshold = self.thresholds["response_time"]
            if avg_time > threshold.critical:
                return PerformanceLevel.CRITICAL
            elif avg_time > threshold.warning:
                return PerformanceLevel.DEGRADED

        # Default classification based on success rate
        if self.success_rate >= 0.95:
            return PerformanceLevel.OPTIMAL
        elif self.success_rate >= 0.85:
            return PerformanceLevel.ACCEPTABLE
        elif self.success_rate >= 0.70:
            return PerformanceLevel.DEGRADED
        else:
            return PerformanceLevel.CRITICAL


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system with metrics collection and analysis.
    """

    def __init__(self, use_prometheus: bool = False, prometheus_port: int = 8000):
        """
        Initialize performance monitor.

        Args:
            use_prometheus: Whether to enable Prometheus metrics
            prometheus_port: Port for Prometheus metrics server
        """
        self.use_prometheus = use_prometheus
        self.prometheus_port = prometheus_port
        self._lock = threading.RLock()
        self.metrics: Dict[str, ComponentMetrics] = {}
        self.historical_data: Dict[str, List[float]] = {}

        # Prometheus metrics
        self.prometheus_counters: Dict[str, Counter] = {}
        self.prometheus_gauges: Dict[str, Gauge] = {}
        self.prometheus_histograms: Dict[str, Histogram] = {}

        if use_prometheus:
            self._init_prometheus()

    def _init_prometheus(self) -> None:
        """Initialize Prometheus metrics server and metrics."""
        try:
            start_http_server(self.prometheus_port)
            logger.info(
                f"Prometheus metrics server started on port {self.prometheus_port}"
            )

            # Create default metrics
            self.prometheus_counters["total_calls"] = Counter(
                "oracle_x_calls_total", "Total number of calls", ["component", "status"]
            )
            self.prometheus_gauges["response_time"] = Gauge(
                "oracle_x_response_time_seconds",
                "Response time in seconds",
                ["component"],
            )
            self.prometheus_histograms["response_time_histogram"] = Histogram(
                "oracle_x_response_time_histogram_seconds",
                "Response time histogram",
                ["component"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            )

        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            self.use_prometheus = False

    def monitor_performance(self, component: str, start_time: float) -> Dict[str, Any]:
        """
        Monitor performance of a component call.

        Args:
            component: Component name
            start_time: Start time of the operation (from time.time())

        Returns:
            Performance metrics dictionary
        """
        end_time = time.time()
        duration = end_time - start_time

        with self._lock:
            # Get or create component metrics
            if component not in self.metrics:
                self.metrics[component] = ComponentMetrics(component=component)
                self.historical_data[component] = []

            metrics = self.metrics[component]
            metrics.total_calls += 1
            metrics.successful_calls += (
                1  # Assume success, caller should call record_failure if needed
            )
            metrics.total_time += duration
            metrics.min_time = min(metrics.min_time, duration)
            metrics.max_time = max(metrics.max_time, duration)
            metrics.last_call_time = datetime.now()

            # Store historical data (keep last 1000 samples)
            historical = self.historical_data[component]
            historical.append(duration)
            if len(historical) > 1000:
                historical.pop(0)

            # Update Prometheus metrics
            if self.use_prometheus:
                self._update_prometheus_metrics(component, duration, "success")

            return {
                "component": component,
                "duration": duration,
                "average_time": metrics.average_time,
                "success_rate": metrics.success_rate,
                "total_calls": metrics.total_calls,
                "performance_level": metrics.current_level.name,
            }

    def record_failure(
        self, component: str, start_time: float, error: Optional[Exception] = None
    ) -> None:
        """
        Record a failed component call.

        Args:
            component: Component name
            start_time: Start time of the operation
            error: Optional exception that caused the failure
        """
        end_time = time.time()
        duration = end_time - start_time

        with self._lock:
            if component not in self.metrics:
                self.metrics[component] = ComponentMetrics(component=component)
                self.historical_data[component] = []

            metrics = self.metrics[component]
            metrics.total_calls += 1
            metrics.failed_calls += 1
            metrics.total_time += duration
            metrics.min_time = min(metrics.min_time, duration)
            metrics.max_time = max(metrics.max_time, duration)
            metrics.last_call_time = datetime.now()

            # Update Prometheus metrics
            if self.use_prometheus:
                self._update_prometheus_metrics(component, duration, "failure")

    def _update_prometheus_metrics(
        self, component: str, duration: float, status: str
    ) -> None:
        """Update Prometheus metrics."""
        try:
            self.prometheus_counters["total_calls"].labels(
                component=component, status=status
            ).inc()

            self.prometheus_gauges["response_time"].labels(component=component).set(
                duration
            )

            self.prometheus_histograms["response_time_histogram"].labels(
                component=component
            ).observe(duration)
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")

    def set_threshold(
        self, component: str, metric: str, threshold: PerformanceThreshold
    ) -> None:
        """
        Set performance threshold for a component.

        Args:
            component: Component name
            metric: Metric name (e.g., 'response_time')
            threshold: Threshold configuration
        """
        with self._lock:
            if component not in self.metrics:
                self.metrics[component] = ComponentMetrics(component=component)
            self.metrics[component].thresholds[metric] = threshold

    def get_metrics(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics.

        Args:
            component: Optional specific component to get metrics for

        Returns:
            Performance metrics dictionary
        """
        with self._lock:
            if component:
                if component in self.metrics:
                    metrics = self.metrics[component]
                    historical = self.historical_data.get(component, [])

                    return {
                        "component": metrics.component,
                        "total_calls": metrics.total_calls,
                        "successful_calls": metrics.successful_calls,
                        "failed_calls": metrics.failed_calls,
                        "success_rate": metrics.success_rate,
                        "average_time": metrics.average_time,
                        "min_time": (
                            metrics.min_time
                            if metrics.min_time != float("inf")
                            else 0.0
                        ),
                        "max_time": metrics.max_time,
                        "total_time": metrics.total_time,
                        "last_call_time": (
                            metrics.last_call_time.isoformat()
                            if metrics.last_call_time
                            else None
                        ),
                        "performance_level": metrics.current_level.name,
                        "percentile_95": (
                            self._calculate_percentile(historical, 95)
                            if historical
                            else 0.0
                        ),
                        "percentile_99": (
                            self._calculate_percentile(historical, 99)
                            if historical
                            else 0.0
                        ),
                        "standard_deviation": (
                            statistics.stdev(historical) if len(historical) > 1 else 0.0
                        ),
                    }
                else:
                    return {}
            else:
                # Return all metrics
                return {comp: self.get_metrics(comp) for comp in self.metrics.keys()}

    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile from data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (len(sorted_data) - 1) * percentile / 100
        lower_index = int(index)
        upper_index = lower_index + 1

        if upper_index >= len(sorted_data):
            return sorted_data[lower_index]

        # Linear interpolation
        lower_value = sorted_data[lower_index]
        upper_value = sorted_data[upper_index]
        return lower_value + (upper_value - lower_value) * (index - lower_index)

    def get_performance_insights(self, component: str) -> Dict[str, Any]:
        """
        Get performance insights and recommendations.

        Args:
            component: Component name

        Returns:
            Performance insights dictionary
        """
        metrics = self.get_metrics(component)
        if not metrics:
            return {}

        insights = {
            "component": component,
            "current_level": metrics["performance_level"],
            "recommendations": [],
            "bottlenecks": [],
        }

        # Generate recommendations based on metrics
        if metrics["success_rate"] < 0.9:
            insights["recommendations"].append(
                f"Improve error handling - current success rate: {metrics['success_rate']:.1%}"
            )

        if metrics["average_time"] > 1000:  # More than 1 second
            insights["recommendations"].append(
                f"Optimize performance - average response time: {metrics['average_time']:.2f}ms"
            )

        if metrics["max_time"] > metrics["average_time"] * 3:
            insights["bottlenecks"].append(
                "Significant performance variance detected - investigate occasional slow requests"
            )

        # Check against thresholds
        comp_metrics = self.metrics.get(component)
        if comp_metrics and "response_time" in comp_metrics.thresholds:
            threshold = comp_metrics.thresholds["response_time"]
            if metrics["average_time"] > threshold.critical:
                insights["bottlenecks"].append(
                    f"Critical performance issue - average time exceeds critical threshold "
                    f"({metrics['average_time']:.2f}{threshold.unit} > {threshold.critical}{threshold.unit})"
                )

        return insights

    def reset_metrics(self, component: Optional[str] = None) -> None:
        """
        Reset performance metrics.

        Args:
            component: Optional specific component to reset
        """
        with self._lock:
            if component:
                if component in self.metrics:
                    self.metrics[component] = ComponentMetrics(component=component)
                    self.historical_data[component] = []
            else:
                self.metrics.clear()
                self.historical_data.clear()

    def export_metrics(self, format: str = "json") -> str:
        """
        Export metrics in specified format.

        Args:
            format: Export format ('json', 'prometheus', 'text')

        Returns:
            Exported metrics string
        """
        metrics = self.get_metrics()

        if format == "json":
            return json.dumps(metrics, indent=2, default=str)
        elif format == "prometheus" and self.use_prometheus:
            # Prometheus format would be handled by the Prometheus client
            return "# Prometheus metrics available at /metrics endpoint"
        else:
            # Text format
            lines = []
            for comp, data in metrics.items():
                lines.append(f"=== {comp} ===")
                lines.append(f"  Total Calls: {data['total_calls']}")
                lines.append(f"  Success Rate: {data['success_rate']:.1%}")
                lines.append(f"  Avg Time: {data['average_time']:.2f}ms")
                lines.append(
                    f"  Min/Max: {data['min_time']:.2f}ms / {data['max_time']:.2f}ms"
                )
                lines.append(f"  Performance: {data['performance_level']}")
                lines.append("")
            return "\n".join(lines)


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(use_prometheus: bool = False) -> PerformanceMonitor:
    """
    Get or create the global performance monitor instance.

    Args:
        use_prometheus: Whether to enable Prometheus metrics

    Returns:
        PerformanceMonitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(use_prometheus=use_prometheus)
    return _performance_monitor


def monitor_decorator(component: str):
    """
    Decorator for monitoring function performance.

    Args:
        component: Component name for monitoring

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                monitor.monitor_performance(component, start_time)
                return result
            except Exception as e:
                monitor.record_failure(component, start_time, e)
                raise

        return wrapper

    return decorator


def monitor_performance(component: str, start_time: float) -> Dict[str, Any]:
    """
    Convenience function for monitoring performance.

    Args:
        component: Component name
        start_time: Start time of the operation

    Returns:
        Performance metrics
    """
    monitor = get_performance_monitor()
    return monitor.monitor_performance(component, start_time)


class PerformanceAlert:
    """Performance alert system for monitoring critical issues."""

    def __init__(self):
        self.alerts: Dict[str, List[Dict]] = {}
        self._lock = threading.RLock()

    def check_alerts(self, component: str, metrics: Dict[str, Any]) -> List[Dict]:
        """
        Check for performance alerts.

        Args:
            component: Component name
            metrics: Performance metrics

        Returns:
            List of alert messages
        """
        alerts = []

        # Check success rate
        if metrics["success_rate"] < 0.8:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "message": f"Low success rate for {component}: {metrics['success_rate']:.1%}",
                    "component": component,
                    "metric": "success_rate",
                    "value": metrics["success_rate"],
                }
            )

        # Check response time
        if metrics["average_time"] > 5000:  # 5 seconds
            alerts.append(
                {
                    "level": "WARNING",
                    "message": f"High response time for {component}: {metrics['average_time']:.2f}ms",
                    "component": component,
                    "metric": "response_time",
                    "value": metrics["average_time"],
                }
            )

        # Store alerts
        with self._lock:
            if component not in self.alerts:
                self.alerts[component] = []
            self.alerts[component].extend(alerts)

        return alerts

    def get_alerts(self, component: Optional[str] = None) -> List[Dict]:
        """
        Get current alerts.

        Args:
            component: Optional specific component

        Returns:
            List of alerts
        """
        with self._lock:
            if component:
                return self.alerts.get(component, [])
            else:
                return [alert for alerts in self.alerts.values() for alert in alerts]

    def clear_alerts(self, component: Optional[str] = None) -> None:
        """
        Clear alerts.

        Args:
            component: Optional specific component
        """
        with self._lock:
            if component:
                self.alerts[component] = []
            else:
                self.alerts.clear()


# Global alert system
_performance_alerts: Optional[PerformanceAlert] = None


def get_performance_alerts() -> PerformanceAlert:
    """
    Get or create the global performance alert system.

    Returns:
        PerformanceAlert instance
    """
    global _performance_alerts
    if _performance_alerts is None:
        _performance_alerts = PerformanceAlert()
    return _performance_alerts
