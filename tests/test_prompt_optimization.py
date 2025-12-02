import sys
import os
import unittest
from unittest.mock import patch, MagicMock, ANY
import json
from datetime import datetime, timedelta
import sqlite3
import tempfile
import random

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from oracle_engine.prompt_optimization import (
        PromptOptimizationEngine,
        MarketCondition,
        PromptStrategy,
        PromptTemplate,
        PromptPerformance,
        get_optimization_engine,
    )
except ImportError:
    # Skip tests if module not available
    print("Skipping prompt_optimization tests - module not available")


class TestPromptOptimizationEngine(unittest.TestCase):
    """Test suite for the LLM self-reflection system (Prompt Optimization Engine)"""

    def setUp(self):
        """Set up test environment with temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db")
        self.engine = PromptOptimizationEngine(db_path=self.temp_db.name)

        # Sample signals for testing
        self.sample_signals = {
            "market_internals": "Market conditions appear bullish with strong momentum.",
            "options_flow": [
                "AAPL call volume increased by 200%",
                "TSLA unusual put activity",
            ],
            "sentiment_llm": "Overall market sentiment is positive with bullish bias.",
            "finviz_breadth": "Market breadth indicators show 70% stocks above MA50.",
            "chart_analysis": "AAPL showing bullish flag pattern with volume confirmation.",
            "earnings_calendar": [
                {"ticker": "AAPL", "date": "2025-07-30", "time": "after_hours"},
                {"ticker": "MSFT", "date": "2025-07-31", "time": "after_hours"},
            ],
        }

    def tearDown(self):
        """Clean up temporary database"""
        self.temp_db.close()

    def test_init_database(self):
        """Test database initialization with correct tables"""
        with sqlite3.connect(self.temp_db.name) as conn:
            # Check if all required tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]

            self.assertIn("prompt_performance", table_names)
            self.assertIn("prompt_experiments", table_names)
            self.assertIn("signal_importance", table_names)

    def test_default_templates(self):
        """Test that default templates are loaded correctly"""
        templates = self.engine.prompt_templates

        # Check that we have expected templates
        self.assertGreaterEqual(len(templates), 4)  # At least 4 default templates
        self.assertIn("conservative_balanced", templates)
        self.assertIn("aggressive_momentum", templates)
        self.assertIn("earnings_specialist", templates)
        self.assertIn("technical_precision", templates)

        # Check template structure
        template = templates["conservative_balanced"]
        self.assertEqual(template.strategy, PromptStrategy.CONSERVATIVE)
        self.assertIn(MarketCondition.SIDEWAYS, template.market_conditions)
        self.assertIn("signals_section", template.user_prompt_template)
        self.assertTrue(len(template.priority_signals) > 0)

    def test_classify_market_condition_bullish(self):
        """Test market condition classification with bullish signals"""
        signals = self.sample_signals.copy()
        signals["sentiment_llm"] = "Very bullish market with strong breakout potential."

        condition = self.engine.classify_market_condition(signals)
        self.assertEqual(condition, MarketCondition.BULLISH)

    def test_classify_market_condition_bearish(self):
        """Test market condition classification with bearish signals"""
        signals = self.sample_signals.copy()
        signals["sentiment_llm"] = (
            "Bearish outlook with negative sentiment and breakdown risk."
        )
        signals["market_internals"] = "Bearish trends with declining volume."

        condition = self.engine.classify_market_condition(signals)
        self.assertEqual(condition, MarketCondition.BEARISH)

    def test_classify_market_condition_earnings(self):
        """Test market condition classification with earnings focus"""
        signals = self.sample_signals.copy()
        # Add more earnings events
        signals["earnings_calendar"] = [
            {"ticker": "AAPL", "date": "2025-07-30", "time": "after_hours"},
            {"ticker": "MSFT", "date": "2025-07-31", "time": "after_hours"},
            {"ticker": "AMZN", "date": "2025-07-31", "time": "after_hours"},
            {"ticker": "GOOG", "date": "2025-08-01", "time": "after_hours"},
        ]

        condition = self.engine.classify_market_condition(signals)
        self.assertEqual(condition, MarketCondition.EARNINGS)

    def test_classify_market_condition_volatile(self):
        """Test market condition classification with volatile signals"""
        signals = self.sample_signals.copy()
        signals["market_internals"] = (
            "Market showing high volatility with erratic swings."
        )
        signals["sentiment_llm"] = "Uncertainty dominates with volatile price action."

        condition = self.engine.classify_market_condition(signals)
        self.assertEqual(condition, MarketCondition.VOLATILE)

    def test_classify_market_condition_options_heavy(self):
        """Test market condition classification with heavy options activity"""
        signals = self.sample_signals.copy()
        # Create a list with more than 20 options flow items
        signals["options_flow"] = [f"Option flow item {i}" for i in range(25)]

        condition = self.engine.classify_market_condition(signals)
        self.assertEqual(condition, MarketCondition.OPTIONS_HEAVY)

    def test_select_optimal_template(self):
        """Test template selection based on market conditions"""
        # Test with bullish condition
        template = self.engine.select_optimal_template(MarketCondition.BULLISH)
        self.assertIn(MarketCondition.BULLISH, template.market_conditions)

        # Test with earnings condition
        template = self.engine.select_optimal_template(MarketCondition.EARNINGS)
        self.assertIn(MarketCondition.EARNINGS, template.market_conditions)

        # Test with performance data
        recent_performance = {
            "technical_precision": {"success_rate": 0.9},
            "conservative_balanced": {"success_rate": 0.5},
        }
        template = self.engine.select_optimal_template(
            MarketCondition.SIDEWAYS, recent_performance
        )
        self.assertEqual(template.template_id, "technical_precision")

    def test_optimize_signal_selection(self):
        """Test signal optimization based on template priorities"""
        template = self.engine.prompt_templates["technical_precision"]

        # Add a large signal that would exceed token budget
        large_signals = self.sample_signals.copy()
        large_signals["low_priority"] = "x" * 10000  # Large low priority signal

        optimized = self.engine.optimize_signal_selection(
            large_signals, template, token_budget=1000
        )

        # Check if high priority signals were kept
        self.assertIn("chart_analysis", optimized)
        self.assertIn("market_internals", optimized)

        # Check if token budget was respected (approximately)
        total_tokens = sum(self.engine._estimate_tokens(v) for v in optimized.values())
        self.assertLessEqual(total_tokens, 1000 * 1.1)  # Allow 10% margin

    def test_generate_optimized_prompt(self):
        """Test optimized prompt generation"""
        system_prompt, user_prompt, metadata = self.engine.generate_optimized_prompt(
            self.sample_signals, market_condition=MarketCondition.BULLISH
        )

        # Check prompt content
        self.assertIsInstance(system_prompt, str)
        self.assertIsInstance(user_prompt, str)
        self.assertIn("signals_section", user_prompt)

        # Check metadata
        self.assertIn("template_id", metadata)
        self.assertIn("market_condition", metadata)
        self.assertIn("strategy", metadata)
        self.assertIn("estimated_tokens", metadata)

    def test_generate_prompt_with_template_id(self):
        """Test prompt generation with specific template ID"""
        system_prompt, user_prompt, metadata = self.engine.generate_optimized_prompt(
            self.sample_signals, template_id="technical_precision"
        )

        self.assertEqual(metadata["template_id"], "technical_precision")
        self.assertEqual(metadata["strategy"], "technical")

    def test_record_prompt_performance(self):
        """Test recording of prompt performance metrics"""
        # Create test metadata
        prompt_metadata = {
            "template_id": "technical_precision",
            "market_condition": "bullish",
            "strategy": "technical",
        }

        # Model attempts with some success/failure
        model_attempts = [
            {"success": True, "latency_sec": 2.5},
            {"success": False, "latency_sec": 3.0},
        ]

        # Record performance
        self.engine.record_prompt_performance(prompt_metadata, model_attempts)

        # Check if it was stored in the database
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.execute(
                "SELECT success_rate, avg_latency FROM prompt_performance WHERE template_id=?",
                ("technical_precision",),
            )
            row = cursor.fetchone()

            # Check calculated metrics
            self.assertEqual(row[0], 0.5)  # 1 out of 2 success
            self.assertEqual(row[1], 2.75)  # Average latency

    def test_start_ab_test(self):
        """Test starting an A/B test experiment"""
        experiment_id = self.engine.start_ab_test(
            "conservative_balanced",
            "aggressive_momentum",
            MarketCondition.SIDEWAYS,
            duration_hours=24,
        )

        # Verify experiment was created
        self.assertIsNotNone(experiment_id)
        self.assertIn(experiment_id, self.engine.active_experiments)

        # Check database entry
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.execute(
                "SELECT variant_a_id, variant_b_id FROM prompt_experiments WHERE experiment_id=?",
                (experiment_id,),
            )
            row = cursor.fetchone()

            self.assertEqual(row[0], "conservative_balanced")
            self.assertEqual(row[1], "aggressive_momentum")

    def test_evolve_prompts(self):
        """Test prompt template evolution"""
        # Use conservative template as parent
        top_performers = ["conservative_balanced"]

        # Generate mutations
        new_templates = self.engine.evolve_prompts(top_performers, mutation_rate=0.5)

        # Check mutations were created
        self.assertGreaterEqual(len(new_templates), 1)

        # Verify mutation creates a new template
        mutated = new_templates[0]
        self.assertNotEqual(mutated.template_id, "conservative_balanced")
        self.assertTrue(mutated.template_id.startswith("conservative_balanced_mut_"))

        # Some properties should be different after mutation
        parent = self.engine.prompt_templates["conservative_balanced"]
        differences = 0

        if mutated.temperature != parent.temperature:
            differences += 1
        if mutated.max_tokens != parent.max_tokens:
            differences += 1
        if mutated.context_compression_ratio != parent.context_compression_ratio:
            differences += 1

        # With mutation_rate=0.5, we expect at least one difference
        self.assertGreater(differences, 0)

    def test_compression_methods(self):
        """Test signal compression functionality"""
        # Test string compression
        long_string = "x" * 1000
        compressed = self.engine._compress_signal(long_string, max_tokens=10)
        self.assertLess(len(compressed), len(long_string))

        # Test list compression
        long_list = list(range(20))
        compressed = self.engine._compress_signal(long_list, max_tokens=10)
        self.assertEqual(len(compressed), 5)  # Should take first 5 items

        # Test dict compression
        long_dict = {f"key{i}": f"value{i}" * 20 for i in range(10)}
        compressed = self.engine._compress_signal(long_dict, max_tokens=10)
        self.assertLessEqual(len(compressed), 3)  # Should take first 3 keys

    def test_format_signals_section(self):
        """Test signals section formatting"""
        template = self.engine.prompt_templates["technical_precision"]
        formatted = self.engine._format_signals_section(self.sample_signals, template)

        # Check proper formatting
        self.assertIn("**Market Internals**", formatted)
        self.assertIn("**Chart Analysis**", formatted)
        self.assertIn("**Options Flow**", formatted)

        # Should include weights
        self.assertIn("(Weight:", formatted)

    def test_get_performance_analytics(self):
        """Test performance analytics collection"""
        # First record some performance data
        prompt_metadata = {
            "template_id": "technical_precision",
            "market_condition": "bullish",
            "strategy": "technical",
        }
        model_attempts = [{"success": True, "latency_sec": 2.0}]
        self.engine.record_prompt_performance(prompt_metadata, model_attempts)

        # Get analytics
        analytics = self.engine.get_performance_analytics(days=1)

        # Check structure
        self.assertIn("template_performance", analytics)
        self.assertIn("experiment_results", analytics)
        self.assertIn("total_templates", analytics)

        # Should have at least one template performance record
        self.assertGreaterEqual(len(analytics["template_performance"]), 1)

    def test_global_instance(self):
        """Test global instance getter function"""
        engine = get_optimization_engine()
        self.assertIsInstance(engine, PromptOptimizationEngine)

        # Should return the same instance on second call
        engine2 = get_optimization_engine()
        self.assertIs(engine, engine2)


if __name__ == "__main__":
    unittest.main()
