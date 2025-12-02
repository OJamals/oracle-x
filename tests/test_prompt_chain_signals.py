#!/usr/bin/env python3
"""
Test suite for prompt_chain.py clean_signals_for_llm function

Tests the ability to clean and summarize signal data before sending to LLM.
"""

import unittest
import pytest
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from oracle_engine.prompt_chain import clean_signals_for_llm
except ImportError as e:
    pytest.skip(f"Could not import module: {e}", allow_module_level=True)


class TestCleanSignalsForLLM(unittest.TestCase):
    """Test suite for clean_signals_for_llm function"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample signals with various data types and formats
        self.test_signals = {
            "market_internals": "SPY +0.5%, QQQ +0.8%, VIX -2.3%, TICK +150, TRIN 0.85, AD Line +800\nMarket breadth positive with 1800 advancers vs 1200 decliners.",
            "options_flow": [
                {
                    "ticker": "AAPL",
                    "contracts": 5000,
                    "sentiment": "bullish",
                    "details": "Jun 24 180 Calls, premium $1.2M",
                },
                {
                    "ticker": "NVDA",
                    "contracts": 3000,
                    "sentiment": "bullish",
                    "details": "Jul 24 200 Calls, premium $900K",
                },
                {
                    "ticker": "MSFT",
                    "contracts": 2000,
                    "sentiment": "bearish",
                    "details": "Jun 24 400 Puts, premium $600K",
                },
                {
                    "ticker": "AMZN",
                    "contracts": 1500,
                    "sentiment": "bullish",
                    "details": "Jun 24 170 Calls, premium $450K",
                },
                {
                    "ticker": "TSLA",
                    "contracts": 4000,
                    "sentiment": "bullish",
                    "details": "Jul 24 250 Calls, premium $1.5M",
                },
                {
                    "ticker": "META",
                    "contracts": 2500,
                    "sentiment": "neutral",
                    "details": "Jun 24 450 Straddle, premium $800K",
                },
                {
                    "ticker": "AMD",
                    "contracts": 1800,
                    "sentiment": "bullish",
                    "details": "Aug 24 140 Calls, premium $550K",
                },
            ],
            "dark_pools": [
                {"ticker": "AAPL", "volume": 2500000, "price": 182.50},
                {"ticker": "NVDA", "volume": 1800000, "price": 450.75},
                {"ticker": "MSFT", "volume": 1200000, "price": 405.25},
                {"ticker": "AMZN", "volume": 900000, "price": 172.30},
                {"ticker": "TSLA", "volume": 1500000, "price": 247.80},
                {"ticker": "META", "volume": 800000, "price": 447.60},
                {"ticker": "AMD", "volume": 600000, "price": 138.90},
            ],
            "sentiment_web": [
                "Positive outlook on tech sector",
                "Concerns about inflation impact",
                "Strong Q2 earnings expectations",
                "Supply chain improvements noted",
                "Retail spending remains strong",
                "Semiconductor demand increasing",
                "AI investments accelerating",
            ],
            "sentiment_llm": "Market sentiment appears cautiously optimistic with focus on technology and AI sectors. Inflation concerns persist but have moderated. Earnings season expected to show resilience.",
            "chart_analysis": "SPY shows bullish pattern with breakout above 50-day moving average. RSI at 65 indicating momentum but not overbought. Support at 445, resistance at 460.",
            "earnings_calendar": [
                {
                    "ticker": "AAPL",
                    "date": "2024-07-30",
                    "time": "AMC",
                    "eps_estimate": 1.45,
                },
                {
                    "ticker": "AMZN",
                    "date": "2024-07-25",
                    "time": "AMC",
                    "eps_estimate": 0.89,
                },
                {
                    "ticker": "MSFT",
                    "date": "2024-07-23",
                    "time": "AMC",
                    "eps_estimate": 2.78,
                },
                {
                    "ticker": "META",
                    "date": "2024-07-24",
                    "time": "AMC",
                    "eps_estimate": 4.65,
                },
                {
                    "ticker": "NVDA",
                    "date": "2024-08-21",
                    "time": "AMC",
                    "eps_estimate": 0.56,
                },
                {
                    "ticker": "TSLA",
                    "date": "2024-07-17",
                    "time": "AMC",
                    "eps_estimate": 0.72,
                },
                {
                    "ticker": "AMD",
                    "date": "2024-07-30",
                    "time": "AMC",
                    "eps_estimate": 0.68,
                },
                {
                    "ticker": "GOOGL",
                    "date": "2024-07-23",
                    "time": "AMC",
                    "eps_estimate": 1.83,
                },
            ],
            "yahoo_headlines": [
                {
                    "headline": "Fed signals potential rate cut in September",
                    "url": "https://example.com/1",
                },
                {
                    "headline": "Tech stocks rally on strong earnings",
                    "url": "https://example.com/2",
                },
                {
                    "headline": "NVIDIA announces new AI chip",
                    "url": "https://example.com/3",
                },
                {
                    "headline": "Apple supplier reports production increase",
                    "url": "https://example.com/4",
                },
                {
                    "headline": "Oil prices drop amid demand concerns",
                    "url": "https://example.com/5",
                },
                {
                    "headline": "Tesla beats delivery estimates",
                    "url": "https://example.com/6",
                },
                {
                    "headline": "Microsoft cloud revenue grows 30%",
                    "url": "https://example.com/7",
                },
            ],
            "finviz_breadth": "S&P 500: 75% above 200-day MA, 68% above 50-day MA\nSector Breadth: Tech (85%), Healthcare (72%), Financials (65%), Energy (45%)",
            "tickers": [
                "AAPL",
                "MSFT",
                "AMZN",
                "NVDA",
                "TSLA",
                "META",
                "GOOGL",
                "AMD",
                "JPM",
                "V",
                "BAC",
                "WMT",
            ],
            # Add empty and very long content for edge case testing
            "empty_field": [],
            "non_list_field": 12345,
            "very_long_text": "A" * 1000
            + " very long text that should be truncated"
            + "Z" * 1000,
        }

    def test_clean_signals_basic(self):
        """Test basic cleaning of signals dictionary"""
        cleaned = clean_signals_for_llm(self.test_signals)

        # Check that the cleaned signals dictionary has expected structure
        self.assertIsInstance(cleaned, dict)

        # Check that all expected fields exist
        expected_fields = [
            "market_internals",
            "options_flow",
            "dark_pools",
            "sentiment_web",
            "sentiment_llm",
            "chart_analysis",
            "earnings_calendar",
            "yahoo_headlines",
            "finviz_breadth",
            "tickers",
        ]

        for field in expected_fields:
            self.assertIn(field, cleaned)

    def test_text_field_truncation(self):
        """Test that long text fields are properly truncated"""
        cleaned = clean_signals_for_llm(self.test_signals)

        # Verify market_internals is truncated to 600 chars max
        self.assertLessEqual(len(cleaned["market_internals"]), 600)

        # Verify sentiment_llm is truncated to 600 chars max
        self.assertLessEqual(len(cleaned["sentiment_llm"]), 600)

        # Verify chart_analysis is truncated to 600 chars max
        self.assertLessEqual(len(cleaned["chart_analysis"]), 600)

        # Verify very_long_text field is not included (as it's not in the expected output fields)
        self.assertNotIn("very_long_text", cleaned)

    def test_list_truncation(self):
        """Test that lists are truncated to max_items"""
        max_items = 5  # Default in the function
        cleaned = clean_signals_for_llm(self.test_signals)

        # Check options_flow is truncated
        self.assertLessEqual(len(cleaned["options_flow"]), max_items)

        # Check dark_pools is truncated
        self.assertLessEqual(len(cleaned["dark_pools"]), max_items)

        # Check sentiment_web is truncated
        self.assertLessEqual(len(cleaned["sentiment_web"]), max_items)

        # Check earnings_calendar is truncated
        self.assertLessEqual(len(cleaned["earnings_calendar"]), max_items)

        # Check yahoo_headlines is truncated
        self.assertLessEqual(len(cleaned["yahoo_headlines"]), max_items)

    def test_custom_max_items(self):
        """Test custom max_items parameter"""
        custom_max = 3
        cleaned = clean_signals_for_llm(self.test_signals, max_items=custom_max)

        # Check options_flow is truncated to custom max
        self.assertLessEqual(len(cleaned["options_flow"]), custom_max)

        # Check dark_pools is truncated to custom max
        self.assertLessEqual(len(cleaned["dark_pools"]), custom_max)

    def test_tickers_truncation(self):
        """Test that tickers field is specially truncated to 10 items max"""
        cleaned = clean_signals_for_llm(self.test_signals)

        # Tickers should be limited to 10 max
        self.assertLessEqual(len(cleaned["tickers"]), 10)

    def test_empty_fields(self):
        """Test handling of empty fields"""
        # Create signals with empty fields
        empty_signals = {
            "market_internals": "",
            "options_flow": [],
            "dark_pools": [],
            "sentiment_web": [],
            "sentiment_llm": "",
            "chart_analysis": "",
            "earnings_calendar": [],
            "yahoo_headlines": [],
            "finviz_breadth": "",
            "tickers": [],
        }

        cleaned = clean_signals_for_llm(empty_signals)

        # Check that empty text fields remain empty strings
        self.assertEqual(cleaned["market_internals"], "")
        self.assertEqual(cleaned["sentiment_llm"], "")
        self.assertEqual(cleaned["chart_analysis"], "")

        # Check that empty lists remain empty lists
        self.assertEqual(cleaned["options_flow"], [])
        self.assertEqual(cleaned["dark_pools"], [])

    def test_non_standard_fields(self):
        """Test handling of non-standard field types"""
        signals = {
            "market_internals": 12345,  # Number instead of string
            "options_flow": "not a list",  # String instead of list
            "dark_pools": {"key": "value"},  # Dict instead of list
            "tickers": 123,  # Number instead of list
        }

        cleaned = clean_signals_for_llm(signals)

        # Check that non-string text fields are converted to strings
        self.assertEqual(cleaned["market_internals"], "12345")

        # Check that non-list fields result in empty lists
        self.assertEqual(cleaned["options_flow"], [])
        self.assertEqual(cleaned["dark_pools"], [])
        self.assertEqual(cleaned["tickers"], [])

    def test_headline_extraction(self):
        """Test proper headline extraction from yahoo_headlines"""
        cleaned = clean_signals_for_llm(self.test_signals)

        # Check that headlines are extracted correctly
        original_headlines = [
            item["headline"] for item in self.test_signals["yahoo_headlines"]
        ]

        # The cleaned data should match a subset of the original headlines
        for item in cleaned["yahoo_headlines"]:
            # Check it's maintaining the dict structure with headline key
            if isinstance(item, dict) and "headline" in item:
                self.assertIn(item["headline"], original_headlines)
            # Or it's been simplified to just the headline text
            elif isinstance(item, str):
                self.assertIn(item, original_headlines)

    def test_yahoo_headlines_deduplication_and_key_handling(self):
        """Ensure yahoo_headlines uses 'headline' key for dedup and respects max_items."""
        signals = {
            "yahoo_headlines": [
                {"headline": "Tech stocks rally on strong earnings", "url": "u1"},
                {
                    "headline": "Tech stocks rally on strong earnings",
                    "url": "u2",
                },  # duplicate headline
                {
                    "headline": "Fed signals potential rate cut in September",
                    "url": "u3",
                },
                {
                    "headline": "Fed signals potential rate cut in September",
                    "url": "u4",
                },  # duplicate headline
                {"headline": "NVIDIA announces new AI chip", "url": "u5"},
                {"headline": "Apple supplier reports production increase", "url": "u6"},
            ]
        }
        cleaned = clean_signals_for_llm(signals, max_items=5)
        # Should not exceed max_items and should be deduplicated by headline text
        self.assertLessEqual(len(cleaned["yahoo_headlines"]), 5)
        headlines_seen = set()
        for item in cleaned["yahoo_headlines"]:
            if isinstance(item, dict) and "headline" in item:
                headline = item["headline"]
            else:
                headline = str(item)
            self.assertNotIn(headline, headlines_seen)
            headlines_seen.add(headline)

    def test_text_cleanup_non_ascii_and_whitespace(self):
        """Verify non-ASCII removal and whitespace collapsing for text fields."""
        signals = {
            "market_internals": " SPY  +0.5%\n\tQQQ  +0.8%  😊  ",
            "sentiment_llm": "Line1\n\nLine2\t\t\nLine3 — emdash",
            "chart_analysis": "Overbought\n\n\nRSI>70\t\t   now",
        }
        cleaned = clean_signals_for_llm(signals)
        # Non-ASCII like emoji or emdash removed, whitespace collapsed to single spaces
        self.assertEqual(cleaned["market_internals"], "SPY +0.5% QQQ +0.8%")
        self.assertNotIn("—", cleaned["sentiment_llm"])  # emdash removed
        self.assertNotRegex(cleaned["sentiment_llm"], r"\s{2,}")
        self.assertNotRegex(cleaned["chart_analysis"], r"\s{2,}")

    def test_list_deduplication_for_simple_lists(self):
        """Ensure simple list fields are deduplicated and truncated."""
        signals = {
            "sentiment_web": [
                "Positive outlook on tech",
                "Positive outlook on tech",  # duplicate
                "Inflation concerns",
                "Inflation concerns",  # duplicate
                "AI investments accelerating",
                "AI investments accelerating",
            ]
        }
        cleaned = clean_signals_for_llm(signals, max_items=3)
        self.assertLessEqual(len(cleaned["sentiment_web"]), 3)
        self.assertEqual(
            len({str(x) for x in cleaned["sentiment_web"]}),
            len(cleaned["sentiment_web"]),
        )

    def test_tickers_trimming_and_dedup(self):
        """Tickers should be deduplicated, stringified if needed, and limited to 10 items."""
        signals = {
            "tickers": [
                "AAPL",
                "TSLA",
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "NVDA",
                "META",
                "AMD",
                "JPM",
                "V",
                "BAC",
                123,
            ]
        }
        cleaned = clean_signals_for_llm(signals)
        self.assertLessEqual(len(cleaned["tickers"]), 10)
        # Ensure no duplicates in the cleaned tickers
        self.assertEqual(
            len(cleaned["tickers"]), len({str(x) for x in cleaned["tickers"]})
        )

    def test_finviz_breadth_truncation(self):
        """finviz_breadth should be truncated to 400 chars with ellipsis when exceeded."""
        long_text = "Breadth:" + "X" * 1000
        signals = {"finviz_breadth": long_text}
        cleaned = clean_signals_for_llm(signals)
        self.assertLessEqual(len(cleaned["finviz_breadth"]), 403)  # 400 + '...'
        self.assertTrue(cleaned["finviz_breadth"].endswith("..."))


if __name__ == "__main__":
    unittest.main()
