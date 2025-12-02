import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from oracle_options_pipeline import create_pipeline, OptionRecommendation


class TestOptionsPortfolioMonitoring(unittest.TestCase):
    def setUp(self):
        self.pipeline = create_pipeline()

    @patch("data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_quote")
    @patch(
        "data_feeds.options_valuation_engine.OptionsValuationEngine.detect_mispricing"
    )
    def test_multiple_positions_action_thresholds(self, mock_detect, mock_quote):
        # Mock a valid underlying quote
        mock_quote_obj = Mock()
        mock_quote_obj.price = 100.0
        mock_quote.return_value = mock_quote_obj

        # Helper to make a valuation mock with market_price
        def make_val(price):
            v = Mock()
            v.market_price = price
            return v

        # entry 2.0 -> current 3.2 = +60% => take_profit
        # entry 5.0 -> current 6.6 = +32% => consider_exit
        # entry 10.0 -> current 7.5 = -25% => stop_loss
        # entry 4.0 -> current 4.6 = +15% => hold
        mock_detect.side_effect = [
            make_val(3.2),  # position 1
            make_val(6.6),  # position 2
            make_val(7.5),  # position 3
            make_val(4.6),  # position 4
        ]

        positions = [
            {
                "symbol": "AAA",
                "strike": 100,
                "expiry": "2025-12-19",
                "type": "call",
                "entry_price": 2.0,
                "quantity": 1,
            },
            {
                "symbol": "BBB",
                "strike": 90,
                "expiry": "2025-11-15",
                "type": "call",
                "entry_price": 5.0,
                "quantity": 2,
            },
            {
                "symbol": "CCC",
                "strike": 110,
                "expiry": "2025-10-17",
                "type": "put",
                "entry_price": 10.0,
                "quantity": 3,
            },
            {
                "symbol": "DDD",
                "strike": 105,
                "expiry": "2025-09-20",
                "type": "put",
                "entry_price": 4.0,
                "quantity": 4,
            },
        ]

        updates = self.pipeline.monitor_positions(positions)

        self.assertEqual(len(updates), 4)
        # Verify actions
        actions = {u["position"]["symbol"]: u["action"] for u in updates}
        self.assertEqual(actions["AAA"], "take_profit")
        self.assertEqual(actions["BBB"], "consider_exit")
        self.assertEqual(actions["CCC"], "stop_loss")
        self.assertEqual(actions["DDD"], "hold")

        # Verify P&L percent and dollar math
        def find(symbol):
            return next(u for u in updates if u["position"]["symbol"] == symbol)

        u1 = find("AAA")
        self.assertAlmostEqual(u1["pnl_percent"], 60.0, places=2)
        self.assertAlmostEqual(u1["pnl_dollar"], (3.2 - 2.0) * 1, places=4)

        u2 = find("BBB")
        self.assertAlmostEqual(u2["pnl_percent"], ((6.6 - 5.0) / 5.0) * 100, places=2)
        self.assertAlmostEqual(u2["pnl_dollar"], (6.6 - 5.0) * 2, places=4)

        u3 = find("CCC")
        self.assertAlmostEqual(u3["pnl_percent"], -25.0, places=2)
        self.assertAlmostEqual(u3["pnl_dollar"], (7.5 - 10.0) * 3, places=4)

        # Timestamp exists and is recent
        for u in updates:
            self.assertIn("timestamp", u)
            self.assertIsInstance(u["timestamp"], datetime)

    @patch("data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_quote")
    @patch(
        "data_feeds.options_valuation_engine.OptionsValuationEngine.detect_mispricing"
    )
    def test_missing_quote_uses_entry_price_and_hold(self, mock_detect, mock_quote):
        # No quote available (e.g., data provider down)
        mock_quote.return_value = None

        positions = [
            {
                "symbol": "ZZZ",
                "strike": 50,
                "expiry": "2025-12-19",
                "type": "call",
                "entry_price": 1.5,
                "quantity": 10,
            },
        ]

        updates = self.pipeline.monitor_positions(positions)
        self.assertEqual(len(updates), 1)
        u = updates[0]
        self.assertEqual(u["current_price"], 1.5)
        self.assertEqual(u["pnl_percent"], 0.0)
        self.assertEqual(u["action"], "hold")
        self.assertIsNone(u["underlying_price"])

        # detect_mispricing should not be called when underlying is None
        mock_detect.assert_not_called()

    @patch("data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_quote")
    @patch(
        "data_feeds.options_valuation_engine.OptionsValuationEngine.detect_mispricing"
    )
    def test_valuation_failure_fallback(self, mock_detect, mock_quote):
        mock_quote_obj = Mock()
        mock_quote_obj.price = 200.0
        mock_quote.return_value = mock_quote_obj
        mock_detect.side_effect = Exception("Valuation failure")

        positions = [
            {
                "symbol": "EEE",
                "strike": 210,
                "expiry": "2025-12-19",
                "type": "call",
                "entry_price": 2.0,
                "quantity": 5,
            },
        ]

        updates = self.pipeline.monitor_positions(positions)
        self.assertEqual(len(updates), 1)
        u = updates[0]
        # Fallback uses entry_price * 1.1 => +10%
        self.assertAlmostEqual(u["current_price"], 2.0 * 1.1, places=6)
        self.assertAlmostEqual(u["pnl_percent"], 10.0, places=6)
        self.assertEqual(u["action"], "hold")

    @patch("data_feeds.data_feed_orchestrator.DataFeedOrchestrator.get_quote")
    def test_malformed_positions_are_skipped(self, mock_quote):
        # Force quote exception to hit error handling path early
        mock_quote.side_effect = Exception("Quote Error")

        positions = [
            {"symbol": "BAD", "strike": 100},  # missing required fields
            {"symbol": "ALSO_BAD"},
        ]

        updates = self.pipeline.monitor_positions(positions)
        # Function should skip malformed entries and return empty list for this case
        self.assertEqual(len(updates), 0)


class TestOptionsPortfolioAnalytics(unittest.TestCase):
    def setUp(self):
        self.pipeline = create_pipeline()

    def test_performance_stats_detailed(self):
        # Prepare cache with recommendations including ml_confidence
        rec1 = Mock(spec=OptionRecommendation)
        rec1.opportunity_score = 80.0
        rec1.ml_confidence = 0.7

        rec2 = Mock(spec=OptionRecommendation)
        rec2.opportunity_score = 60.0
        rec2.ml_confidence = 0.9

        rec3 = Mock(spec=OptionRecommendation)
        rec3.opportunity_score = 100.0
        rec3.ml_confidence = None  # ensure None is ignored in avg

        self.pipeline._cache["AAPL"] = [rec1, rec3]
        self.pipeline._cache["TSLA"] = [rec2]

        stats = self.pipeline.get_performance_stats()

        self.assertEqual(stats["cache_size"], 2)
        self.assertEqual(stats["total_recommendations"], 3)
        # Average opportunity score: (80 + 60 + 100) / 3
        self.assertAlmostEqual(
            stats["avg_opportunity_score"], (80 + 60 + 100) / 3.0, places=6
        )
        # Average ML confidence over non-None values: (0.7 + 0.9) / 2
        self.assertAlmostEqual(stats["avg_ml_confidence"], (0.7 + 0.9) / 2.0, places=6)

        # Top symbols sorting by count
        top = stats["top_symbols"]
        self.assertTrue(("AAPL", 2) in top and ("TSLA", 1) in top)

        # Cache hit rate metric as defined (active_cache_entries / total_cache_entries)
        # both entries have data -> 2 / 2 = 1.0
        self.assertAlmostEqual(stats["cache_hit_rate"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
