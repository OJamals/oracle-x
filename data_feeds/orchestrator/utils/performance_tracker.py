"""
PerformanceTracker - Tracks data source performance for DataFeedOrchestrator
Extracted from monolithic data_feed_orchestrator.py for modularity.
"""

import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks data source performance and reliability with issue registry."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(self._create_metrics_dict)
        self.issues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    @staticmethod
    def _create_metrics_dict():
        return {
            "response_times": deque(maxlen=100),
            "success_count": 0,
            "error_count": 0,
            "quality_scores": deque(maxlen=100),
            "last_success": None,
            "last_error": None,
            "issues": deque(maxlen=50),
        }

    def record_success(
        self,
        source: str,
        response_time: float,
        quality_score: float,
        issues: Optional[List[str]] = None,
    ):
        metrics = self.metrics[source]
        metrics["response_times"].append(response_time)
        metrics["success_count"] += 1
        metrics["quality_scores"].append(quality_score)
        metrics["last_success"] = datetime.now()
        if issues:
            for issue in issues:
                self.record_issue(source, issue)

    def record_error(self, source: str, error: str):
        metrics = self.metrics[source]
        metrics["error_count"] += 1
        metrics["last_error"] = datetime.now()
        logger.warning(f"Data source {source} error: {error}")
        self.record_issue(source, f"ERROR: {error}")

    def record_issue(self, source: str, issue: str):
        try:
            self.metrics[source]["issues"].append(issue)
            self.issues[source].append(issue)
        except Exception:
            pass

    def get_source_ranking(self) -> List[Tuple[str, float]]:
        rankings: List[Tuple[str, float]] = []
        for source, metrics in self.metrics.items():
            avg_quality = (
                (sum(metrics["quality_scores"]) / len(metrics["quality_scores"]))
                if metrics["quality_scores"]
                else 0.0
            )
            rankings.append((source, avg_quality))
        return sorted(rankings, key=lambda x: x[1], reverse=True)
