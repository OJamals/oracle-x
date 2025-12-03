"""
DataValidator - Quality validation for DataFeedOrchestrator data sources
Extracted from monolithic data_feed_orchestrator.py for modularity.
"""

from typing import List, Optional, Tuple

import pandas as pd
from datetime import datetime

from data_feeds.data_types import Quote


class DataValidator:
    """Optimized data validation and anomaly detection"""

    @staticmethod
    def validate_quote(quote: Quote) -> Tuple[float, List[str]]:
        """Fast quote validation with early exits"""
        issues = []
        score = 100.0

        # Price validation - most critical
        if not quote.price or quote.price <= 0:
            issues.append("Missing or invalid price")
            score -= 40
            return max(0, score), issues  # Early exit for critical failure

        # Volume validation
        if not quote.volume or quote.volume < 0:
            issues.append("Missing or invalid volume")
            score -= 20

        # Timestamp validation - optimized check
        if quote.timestamp:
            age_seconds = (datetime.now() - quote.timestamp).total_seconds()
            if age_seconds > 3600:  # 1 hour
                issues.append("Stale quote timestamp")
                score -= 20
        else:
            issues.append("Missing timestamp")
            score -= 10

        # Range validation - only if all values present
        if (
            quote.day_low is not None
            and quote.day_high is not None
            and quote.price is not None
        ):
            if not (quote.day_low <= quote.price <= quote.day_high):
                issues.append("Price outside day range")
                score -= 10

        return max(0, min(score, 100)), issues

    @staticmethod
    def validate_market_data(data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Optimized market data validation using pandas vectorization"""
        issues = []
        score = 100.0

        if data.empty:
            return 0.0, ["Empty DataFrame"]

        # Fast missing data calculation
        total_cells = data.size
        if total_cells > 0:
            missing_cells = data.isnull().sum().sum()
            missing_pct = (missing_cells / total_cells) * 100
            if missing_pct > 5:
                issues.append(f"Missing data: {missing_pct:.2f}%")
                score -= 20

        # Vectorized outlier detection for numeric columns
        numeric_cols = data.select_dtypes(include=[float, int]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                vals = data[col].dropna()
                if len(vals) > 2:
                    # Fast vectorized z-score calculation
                    mean_val = vals.mean()
                    std_val = vals.std()
                    if std_val > 0:
                        z_scores = ((vals - mean_val) / std_val).abs()
                        if (z_scores > 5).any():
                            issues.append(f"Outlier detected in {col}")
                            score -= 10
                            break  # Don't check all columns if one has outliers

        return max(0, min(score, 100)), issues

    @staticmethod
    def detect_anomalies(data: pd.Series, threshold: float = 3.0) -> List[int]:
        """Vectorized anomaly detection"""
        if data.empty:
            return []

        vals = data.dropna()
        if len(vals) < 2:
            return []

        # Vectorized z-score calculation
        mean_val = vals.mean()
        std_val = vals.std()

        if std_val == 0:
            return []

        z_scores = ((vals - mean_val) / std_val).abs()
        anomalies = vals[z_scores > threshold]

        return anomalies.index.tolist()
