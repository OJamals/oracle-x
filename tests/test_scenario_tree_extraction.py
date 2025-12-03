#!/usr/bin/env python3
"""
Test suite for scenario tree extraction functions in prompt_chain.py

Tests the extraction of scenario trees from various LLM output formats.
"""

import unittest
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from oracle_engine.chains.prompt_chain import (
        extract_scenario_tree,
        _extracted_from_extract_scenario_tree_42,
    )
except ImportError as e:
    pytest.skip(f"Could not import module: {e}", allow_module_level=True)


class TestScenarioTreeExtraction(unittest.TestCase):
    """Test suite for scenario tree extraction functionality"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample valid JSON output with scenario_tree
        self.valid_json = """{
            "scenario_tree": {
                "base_case": "60% - Market trades sideways with low volatility.",
                "bull_case": "25% - Fed signals rate pause, market rallies.",
                "bear_case": "15% - Inflation data disappoints, market sells off."
            }
        }"""

        # Create sample markdown with code block containing JSON
        self.markdown_json = """
        Here's my analysis:

        ```json
        {
            "scenario_tree": {
                "base_case": "50% - AAPL consolidates near support.",
                "bull_case": "30% - AAPL breaks resistance on volume.",
                "bear_case": "20% - AAPL breaks down on weak earnings."
            }
        }
        ```

        Let me know if you need anything else.
        """

        # Create sample output with scenario_tree as a dict pattern
        self.dict_pattern = """
        The market analysis suggests the following probabilities:

        "scenario_tree": {
            "base_case": "55% - Range-bound trading continues.",
            "bull_case": "25% - Breakout above resistance.",
            "bear_case": "20% - Breakdown below support."
        }

        These probabilities reflect current market conditions.
        """

        # Create sample output with scenario_tree in a code block
        self.code_block = """
        Based on my analysis:

        ```python
        result = {
            "scenario_tree": {
                "base_case": "65% - Stock trades sideways within range.",
                "bull_case": "20% - Breaks out on positive news.",
                "bear_case": "15% - Breaks down on market weakness."
            }
        }
        ```

        This reflects the current technical setup.
        """

        # Create sample with trade list containing scenario_tree
        self.trade_list = """{
            "trades": [
                {
                    "ticker": "AAPL",
                    "direction": "long",
                    "scenario_tree": {
                        "base_case": "70% - Continues up on strong demand.",
                        "bull_case": "20% - Explosive move if market rallies.",
                        "bear_case": "10% - Drops if market reverses."
                    }
                }
            ]
        }"""

        # Create sample fallback dict-like output (when strict=False)
        self.dict_like = """
        Here's what I think:

        {
            "base_case": "60% - Trades flat",
            "bull_case": "25% - Moves higher",
            "bear_case": "15% - Drops lower"
        }
        """

        # Invalid or edge cases
        self.malformed_json = """{
            "scenario_tree": {
                "base_case": "60% - Market trades sideways,
                "bull_case": 25,
                "bear_case": "15% - Inflation data disappoints."
            }
        """

        self.empty_string = ""

        self.no_scenario_tree = """
        {
            "market_analysis": "The market looks positive.",
            "recommendation": "Consider buying tech stocks."
        }
        """

    def test_extract_valid_direct_json(self):
        """Test extraction from valid direct JSON"""
        result = extract_scenario_tree(self.valid_json)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertIn("base_case", result)
        self.assertIn("bull_case", result)
        self.assertIn("bear_case", result)
        self.assertEqual(
            result["base_case"], "60% - Market trades sideways with low volatility."
        )

    def test_extract_from_markdown_json(self):
        """Test extraction from JSON in markdown code block"""
        result = extract_scenario_tree(self.markdown_json)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertIn("base_case", result)
        self.assertEqual(result["base_case"], "50% - AAPL consolidates near support.")

    def test_extract_from_dict_pattern(self):
        """Test extraction using regex dict pattern"""
        result = extract_scenario_tree(self.dict_pattern)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertIn("base_case", result)
        self.assertEqual(result["base_case"], "55% - Range-bound trading continues.")

    def test_extract_from_code_block(self):
        """Test extraction from generic code block"""
        result = extract_scenario_tree(self.code_block)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertIn("base_case", result)
        self.assertEqual(
            result["base_case"], "65% - Stock trades sideways within range."
        )

    def test_extract_from_trade_list(self):
        """Test extraction from trades list containing scenario_tree"""
        result = extract_scenario_tree(self.trade_list)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertIn("base_case", result)
        self.assertEqual(result["base_case"], "70% - Continues up on strong demand.")

    def test_fallback_extraction(self):
        """Test fallback extraction of dict-like structures when strict=False"""
        # Should succeed with strict=False (default)
        result = extract_scenario_tree(self.dict_like)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertIn("base_case", result)
        self.assertEqual(result["base_case"], "60% - Trades flat")

        # Should fail with strict=True
        result = extract_scenario_tree(self.dict_like, strict=True)
        self.assertIsNone(result)

    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        result = extract_scenario_tree(self.malformed_json)
        self.assertIsNone(result)

    def test_empty_string(self):
        """Test handling of empty string"""
        result = extract_scenario_tree(self.empty_string)
        self.assertIsNone(result)

    def test_no_scenario_tree(self):
        """Test handling of JSON without scenario_tree"""
        result = extract_scenario_tree(self.no_scenario_tree)
        self.assertIsNone(result)

    def test_extracted_from_extract_helper(self):
        """Test the helper function that extracts from regex matches"""
        import re

        # Create a regex match object
        dict_pattern = r'"scenario_tree"\s*:\s*({.*?})'
        match = re.search(dict_pattern, self.dict_pattern, re.DOTALL)

        result = _extracted_from_extract_scenario_tree_42(match, "Test debug message")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertIn("base_case", result)
        self.assertEqual(result["base_case"], "55% - Range-bound trading continues.")

    def test_complex_nested_json(self):
        """Test extraction from complex nested JSON structure"""
        complex_json = """{
            "market_analysis": {
                "summary": "Mixed signals in the market",
                "technical": {
                    "scenario_tree": {
                        "base_case": "45% - Market consolidates at current levels",
                        "bull_case": "30% - Market tests previous highs",
                        "bear_case": "25% - Market tests recent lows"
                    }
                },
                "fundamental": "Strong earnings season expected"
            }
        }"""

        # This should not find the scenario_tree because it's not at the expected paths
        result = extract_scenario_tree(complex_json)
        self.assertIsNone(result)

    def test_quotes_handling(self):
        """Test handling of different quote styles in JSON"""
        single_quotes = """{
            'scenario_tree': {
                'base_case': '55% - Market continues sideways',
                'bull_case': '25% - Market breaks resistance',
                'bear_case': '20% - Market breaks support'
            }
        }"""

        result = extract_scenario_tree(single_quotes)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["base_case"], "55% - Market continues sideways")


if __name__ == "__main__":
    unittest.main()
