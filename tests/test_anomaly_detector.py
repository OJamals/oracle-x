#!/usr/bin/env python3
"""
Test suite for anomaly_detector.py

Tests the ability to detect volume/price anomalies in market data.
"""

import unittest
import pytest
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from data_feeds.anomaly_detector import (
        detect_price_volume_anomalies,
        _as_numeric_series,
    )
except ImportError as e:
    pytest.skip(f"Could not import module: {e}", allow_module_level=True)


class TestAnomalyDetector(unittest.TestCase):
    """Test suite for anomaly detection functionality"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for anomaly detection tests

        # Time series with known anomalies (at indices 4 and 8)
        self.standard_series = pd.Series(
            [100, 105, 95, 110, 300, 105, 98, 100, 350, 103]
        )

        # DataFrame with volume column containing anomalies
        self.test_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=10),
                "price": [
                    150.0,
                    152.0,
                    151.5,
                    153.0,
                    155.0,
                    154.0,
                    153.5,
                    152.5,
                    153.0,
                    154.5,
                ],
                "volume": [
                    1000000,
                    1200000,
                    950000,
                    1100000,
                    5000000,
                    1050000,
                    980000,
                    1000000,
                    4500000,
                    1030000,
                ],
            }
        )

        # List of dictionaries with volume data
        self.volume_dicts = [
            {"date": "2023-01-01", "volume": 1000000, "price": 150.0},
            {"date": "2023-01-02", "volume": 1200000, "price": 152.0},
            {"date": "2023-01-03", "volume": 950000, "price": 151.5},
            {"date": "2023-01-04", "volume": 1100000, "price": 153.0},
            {"date": "2023-01-05", "volume": 5000000, "price": 155.0},  # Anomaly
            {"date": "2023-01-06", "volume": 1050000, "price": 154.0},
            {"date": "2023-01-07", "volume": 980000, "price": 153.5},
            {"date": "2023-01-08", "volume": 1000000, "price": 152.5},
            {"date": "2023-01-09", "volume": 4500000, "price": 153.0},  # Anomaly
            {"date": "2023-01-10", "volume": 1030000, "price": 154.5},
        ]

        # Simple list of numeric values
        self.numeric_list = [100, 105, 95, 110, 300, 105, 98, 100, 350, 103]

        # Edge cases
        self.empty_series = pd.Series([])
        self.constant_series = pd.Series([100, 100, 100, 100, 100])
        self.small_series = pd.Series([100, 110])
        self.series_with_nans = pd.Series([100, np.nan, 105, 110, np.nan, 300])

    def test_numeric_series_conversion(self):
        """Test conversion of various data types to numeric Series"""

        # Test pandas Series conversion
        series_result = _as_numeric_series(self.standard_series)
        self.assertIsInstance(series_result, pd.Series)
        self.assertEqual(len(series_result), 10)

        # Test DataFrame with volume column
        df_result = _as_numeric_series(self.test_df)
        self.assertIsInstance(df_result, pd.Series)
        self.assertEqual(len(df_result), 10)
        # Check it used the volume column
        self.assertEqual(df_result.iloc[4], 5000000)

        # Test list of dictionaries with volume
        dict_result = _as_numeric_series(self.volume_dicts)
        self.assertIsInstance(dict_result, pd.Series)
        self.assertEqual(len(dict_result), 10)
        self.assertEqual(dict_result.iloc[4], 5000000)

        # Test simple numeric list
        list_result = _as_numeric_series(self.numeric_list)
        self.assertIsInstance(list_result, pd.Series)
        self.assertEqual(len(list_result), 10)
        self.assertEqual(list_result.iloc[4], 300)

    def test_numeric_series_edge_cases(self):
        """Test edge cases in numeric series conversion"""

        # Test empty series - should return empty series, not raise error
        empty_result = _as_numeric_series(self.empty_series)
        self.assertIsInstance(empty_result, pd.Series)
        self.assertEqual(len(empty_result), 0)

        # Test handling of NaN values
        nan_result = _as_numeric_series(self.series_with_nans)
        self.assertIsInstance(nan_result, pd.Series)
        self.assertEqual(len(nan_result), 4)  # Should have dropped the two NaN values

        # Test invalid input types
        with self.assertRaises(TypeError):
            _as_numeric_series("not a valid input")

        with self.assertRaises(TypeError):
            _as_numeric_series(123)

        # Test DataFrame with no suitable columns
        invalid_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5),
                "category": ["A", "B", "C", "D", "E"],
            }
        )
        with self.assertRaises(ValueError):
            _as_numeric_series(invalid_df)

        # Test list of dictionaries with no volume key
        invalid_dicts = [
            {"date": "2023-01-01", "price": 150.0},
            {"date": "2023-01-02", "price": 152.0},
        ]
        with self.assertRaises(ValueError):
            _as_numeric_series(invalid_dicts)

    def test_anomaly_detection_series(self):
        """Test anomaly detection with pandas Series input"""
        # Default threshold (3.0)
        anomalies = detect_price_volume_anomalies(self.standard_series)
        self.assertEqual(
            len(anomalies), 0
        )  # With threshold 3.0, no anomalies should be detected

        # Lower threshold to catch our known anomalies
        anomalies = detect_price_volume_anomalies(self.standard_series, threshold=2.0)
        self.assertEqual(
            len(anomalies), 1
        )  # Only the most extreme anomaly (350) exceeds threshold 2.0
        self.assertIn(8, anomalies)  # Index 8 has value 350 with Z-score > 2.0

    def test_anomaly_detection_dataframe(self):
        """Test anomaly detection with DataFrame input"""
        # Default threshold (3.0)
        anomalies = detect_price_volume_anomalies(self.test_df)
        self.assertEqual(
            len(anomalies), 0
        )  # With threshold 3.0, no anomalies should be detected

        # Lower threshold to catch our known anomalies
        anomalies = detect_price_volume_anomalies(self.test_df, threshold=2.0)
        self.assertEqual(
            len(anomalies), 1
        )  # Only the most extreme anomaly (5000000) exceeds threshold 2.0
        self.assertIn(4, anomalies)  # Index 4 has volume 5000000 with Z-score > 2.0

    def test_anomaly_detection_dict_list(self):
        """Test anomaly detection with list of dictionaries input"""
        # Default threshold (3.0)
        anomalies = detect_price_volume_anomalies(self.volume_dicts)
        self.assertEqual(
            len(anomalies), 0
        )  # With threshold 3.0, no anomalies should be detected

        # Lower threshold to catch our known anomalies
        anomalies = detect_price_volume_anomalies(self.volume_dicts, threshold=2.0)
        self.assertEqual(
            len(anomalies), 1
        )  # Only the most extreme anomaly (5000000) exceeds threshold 2.0
        self.assertIn(4, anomalies)  # Index 4 has volume 5000000 with Z-score > 2.0

    def test_anomaly_detection_numeric_list(self):
        """Test anomaly detection with simple numeric list input"""
        # Default threshold (3.0)
        anomalies = detect_price_volume_anomalies(self.numeric_list)
        self.assertEqual(
            len(anomalies), 0
        )  # With threshold 3.0, no anomalies should be detected

        # Lower threshold to catch our known anomalies
        anomalies = detect_price_volume_anomalies(self.numeric_list, threshold=2.0)
        self.assertEqual(
            len(anomalies), 1
        )  # Only the most extreme anomaly (350) exceeds threshold 2.0
        self.assertIn(8, anomalies)  # Index 8 has value 350 with Z-score > 2.0

    def test_anomaly_detection_edge_cases(self):
        """Test anomaly detection edge cases"""
        # Empty series should return empty result
        self.assertEqual(detect_price_volume_anomalies(self.empty_series), [])

        # Series too short should return empty result
        self.assertEqual(detect_price_volume_anomalies(self.small_series), [])

        # Constant series (zero std dev) should return empty result
        self.assertEqual(detect_price_volume_anomalies(self.constant_series), [])

        # Series with NaNs should work on the clean values
        nan_anomalies = detect_price_volume_anomalies(
            self.series_with_nans, threshold=1.5
        )
        self.assertEqual(len(nan_anomalies), 1)  # Should detect one anomaly (300)

    def test_real_world_example(self):
        """Test with realistic market data pattern"""
        # Create more realistic volume data with weekly patterns and occasional spikes
        dates = pd.date_range(start="2023-01-01", periods=30)

        # Base volume with weekly pattern (higher mid-week)
        base_volumes = []
        for i in range(30):
            day_of_week = i % 7
            if day_of_week < 5:  # Weekday
                # Higher volume mid-week
                base_volume = 1000000 + 200000 * (3 - abs(day_of_week - 2))
            else:  # Weekend
                base_volume = 700000
            base_volumes.append(base_volume)

        # Add random variation
        np.random.seed(42)  # For reproducibility
        random_factors = np.random.normal(1.0, 0.1, 30)
        volumes = [int(v * f) for v, f in zip(base_volumes, random_factors)]

        # Add anomalies
        volumes[10] = 3000000  # Earnings announcement
        volumes[20] = 3500000  # Product launch

        # Create DataFrame
        market_data = pd.DataFrame({"date": dates, "volume": volumes})

        # Detect anomalies
        anomalies = detect_price_volume_anomalies(market_data, threshold=2.0)

        # We should detect the two known anomalies
        self.assertEqual(len(anomalies), 2)
        self.assertIn(10, anomalies)
        self.assertIn(20, anomalies)

        # With higher threshold, we should detect fewer or no anomalies
        strict_anomalies = detect_price_volume_anomalies(market_data, threshold=3.0)
        self.assertLessEqual(len(strict_anomalies), 2)


if __name__ == "__main__":
    unittest.main()
