"""
Helper utilities extracted from data_feed_orchestrator.py
"""

from typing import Any, List, Optional
from datetime import datetime
from decimal import Decimal


# Cached datetime formats for faster parsing
_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
)


def _to_decimal(val: Any) -> Optional[Decimal]:
    """Optimized decimal conversion with early exits"""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val
    if isinstance(val, (int, float)):
        try:
            return Decimal(val)
        except Exception:
            return None
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() in ("none", "null", "nan", ""):
            return None
        try:
            return Decimal(val)
        except Exception:
            return None
    return None


def _parse_datetime(val: Any) -> Optional[datetime]:
    """Optimized datetime parsing with cached formats"""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(float(val))
        except (ValueError, OSError):
            return None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        # Try cached formats in order of frequency
        for fmt in _DATETIME_FORMATS:
            try:
                return datetime.strptime(val, fmt)
            except ValueError:
                continue
    return None


def _batch_to_decimal(values: List[Any]) -> List[Optional[Decimal]]:
    """Vectorized decimal conversion for batch processing"""
    return [_to_decimal(val) for val in values]


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Fast float conversion with fallback"""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() in ("none", "null", "nan", ""):
            return default
        try:
            return float(val)
        except ValueError:
            return default
    return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Fast integer conversion with fallback"""
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() in ("none", "null", "nan", ""):
            return default
        try:
            return int(float(val))  # Handle strings like "123.0"
        except ValueError:
            return default
    return default


def _log_error_and_record(
    perf, source: str, msg: str, exc: Exception
):  # Forward ref to PerformanceTracker
    """Log error and record in performance tracker"""
    emsg = f"{msg}: {exc}"
    logger = logging.getLogger(__name__)
    logger.error(emsg)
    try:
        perf.record_error(source, emsg)
    except Exception:
        pass
