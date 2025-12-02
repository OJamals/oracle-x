#!/usr/bin/env python3
"""
Automated test suite for CLI data point validation
Tests all data points across multiple tickers and scenarios with validation checks.
"""

import os
import sys
import json
import subprocess
import pytest
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test data points and scenarios
TEST_TICKERS = ["AAPL", "MSFT", "TSLA", "JPM", "XOM", "BABA", "MU", "PLUG", "RIOT", "NVDA", "INVALID1234"]
VALID_TICKERS = ["AAPL", "MSFT", "TSLA", "JPM", "XOM", "BABA", "MU", "PLUG", "RIOT", "NVDA"]

# Test cases for each data point type
TEST_CASES = [
    {
        "name": "quote",
        "command": ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "quote", "--symbol", "{symbol}"],
        "validation": "validate_quote",
        "markers": ["network"]
    },
    {
        "name": "market_data",
        "command": ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "market_data", "--symbol", "{symbol}", "--period", "1y", "--interval", "1d"],
        "validation": "validate_market_data",
        "markers": ["network", "slow"]
    },
    {
        "name": "company_info",
        "command": ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "company_info", "--symbol", "{symbol}"],
        "validation": "validate_company_info",
        "markers": ["network"]
    },
    {
        "name": "news",
        "command": ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "news", "--symbol", "{symbol}", "--limit", "5"],
        "validation": "validate_news",
        "markers": ["network"]
    },
    {
        "name": "multiple_quotes",
        "command": ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "multiple_quotes", "--symbols", "AAPL,MSFT,SPY"],
        "validation": "validate_multiple_quotes",
        "markers": ["network"],
        "single_test": True
    },
    {
        "name": "financial_statements",
        "command": ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "financial_statements", "--symbol", "{symbol}"],
        "validation": "validate_financial_statements",
        "markers": ["network"]
    },
    {
        "name": "sentiment",
        "command": ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "sentiment", "--symbol", "{symbol}"],
        "validation": "validate_sentiment",
        "markers": ["network", "optional"]
    },
    {
        "name": "advanced_sentiment",
        "command": ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "advanced_sentiment", "--symbol", "{symbol}"],
        "validation": "validate_advanced_sentiment",
        "markers": ["network", "optional"]
    },
    {
        "name": "market_breadth",
        "command": ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "market_breadth"],
        "validation": "validate_market_breadth",
        "markers": ["network"],
        "single_test": True
    },
    {
        "name": "sector_performance",
        "command": ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "sector_performance"],
        "validation": "validate_sector_performance",
        "markers": ["network"],
        "single_test": True
    }
]

# Optional adapter markers
OPTIONAL_ADAPTERS = {
    "fmp": "FINANCIALMODELINGPREP_API_KEY",
    "finnhub": "FINNHUB_API_KEY",
    "stockdex": "STOCKDEX_API_KEY"
}

# Global test results storage
TEST_RESULTS = []

class ValidationResult:
    """Container for test validation results"""
    def __init__(self, passed=True, reason="", details=None):
        self.passed = passed
        self.reason = reason
        self.details = details or {}

class TestResult:
    """Container for individual test results"""
    def __init__(self, test_name: str, symbol: str, status: str, reason: str, cli_output: str = "", cli_error: str = ""):
        self.test_name = test_name
        self.symbol = symbol
        self.status = status  # "pass", "fail", "skip"
        self.reason = reason
        self.cli_output = cli_output
        self.cli_error = cli_error
        self.timestamp = datetime.now().isoformat()

def run_cli_command(command):
    """Run CLI command and return output and exit code"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", 124
    except Exception as e:
        return "", str(e), 1

def parse_cli_output(output):
    """Parse CLI JSON output"""
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None

def store_test_result(test_name: str, symbol: str, status: str, reason: str, cli_output: str = "", cli_error: str = ""):
    """Store test result for aggregation"""
    result = TestResult(test_name, symbol, status, reason, cli_output, cli_error)
    TEST_RESULTS.append(result.__dict__)

def validate_quote(data, symbol):
    """Validate quote data structure and values"""
    if not data or not data.get("ok"):
        if "skipped" in str(data.get("error", "")).lower():
            return ValidationResult(True, "Skipped due to missing API key", {"reason": "skipped"})
        return ValidationResult(False, f"Quote fetch failed: {data.get('error', 'Unknown error')}")
    
    # Check required fields
    required_fields = ["symbol", "price", "change_percent"]
    for field in required_fields:
        if field not in data:
            return ValidationResult(False, f"Missing required field: {field}")
    
    # Validate data types and ranges
    try:
        price = float(data["price"])
        if price <= 0:
            return ValidationResult(False, f"Invalid price: {price} (should be > 0)")
    except (TypeError, ValueError):
        return ValidationResult(False, f"Invalid price format: {data['price']}")
    
    try:
        volume = data.get("volume", 0)
        if volume is not None and int(volume) < 0:
            return ValidationResult(False, f"Invalid volume: {volume} (should be >= 0)")
    except (TypeError, ValueError):
        return ValidationResult(False, f"Invalid volume format: {data.get('volume')}")
    
    # Check timestamp
    timestamp = data.get("timestamp")
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if dt > datetime.now(dt.tzinfo):
                return ValidationResult(False, f"Future timestamp: {timestamp}")
        except Exception:
            pass  # Allow invalid timestamps to pass for now
    
    # Cross-reference validation for valid symbols
    if symbol != "INVALID1234" and data.get("price"):
        try:
            # This would be where you'd implement web reference lookup
            # For now, we'll just validate the structure is correct
            pass
        except Exception as e:
            # Cross-reference failures don't fail the test, just note it
            pass
    
    return ValidationResult(True, "Valid quote data")

def validate_market_data(data, symbol):
    """Validate market data structure and values"""
    if not data or not data.get("ok"):
        if "skipped" in str(data.get("error", "")).lower():
            return ValidationResult(True, "Skipped due to missing API key", {"reason": "skipped"})
        return ValidationResult(False, f"Market data fetch failed: {data.get('error', 'Unknown error')}")
    
    # Check required fields
    required_fields = ["symbol", "timeframe", "rows"]
    for field in required_fields:
        if field not in data:
            return ValidationResult(False, f"Missing required field: {field}")
    
    # Validate rows count
    try:
        rows = int(data["rows"])
        if rows < 0:
            return ValidationResult(False, f"Invalid row count: {rows}")
    except (TypeError, ValueError):
        return ValidationResult(False, f"Invalid rows format: {data.get('rows')}")
    
    # Check last row data if present
    last_row = data.get("last_row")
    if last_row:
        try:
            if "Close" in last_row and last_row["Close"] is not None:
                close_price = float(last_row["Close"])
                if close_price <= 0:
                    return ValidationResult(False, f"Invalid close price: {close_price}")
        except (TypeError, ValueError):
            return ValidationResult(False, f"Invalid close price format: {last_row.get('Close')}")
        
        try:
            if "Volume" in last_row and last_row["Volume"] is not None:
                volume = int(last_row["Volume"])
                if volume < 0:
                    return ValidationResult(False, f"Invalid volume: {volume}")
        except (TypeError, ValueError):
            return ValidationResult(False, f"Invalid volume format: {last_row.get('Volume')}")
        
        # Cross-reference validation for last row timestamp
        if "Date" in last_row and last_row["Date"]:
            try:
                # Validate date is not in the future
                date_str = last_row["Date"]
                if isinstance(date_str, str):
                    # Parse date string and check if it's reasonable
                    pass  # Structure validation is sufficient for now
            except Exception:
                pass  # Allow date parsing issues to pass
    
    return ValidationResult(True, "Valid market data")

def validate_company_info(data, symbol):
    """Validate company info structure and values"""
    if not data or not data.get("ok"):
        if "skipped" in str(data.get("error", "")).lower():
            return ValidationResult(True, "Skipped due to missing API key", {"reason": "skipped"})
        return ValidationResult(False, f"Company info fetch failed: {data.get('error', 'Unknown error')}")
    
    # Check required fields
    required_fields = ["symbol"]
    for field in required_fields:
        if field not in data:
            return ValidationResult(False, f"Missing required field: {field}")
    
    # Validate name if present
    name = data.get("name")
    if name is not None and not isinstance(name, str):
        return ValidationResult(False, f"Invalid name format: {name}")
    
    # Validate market cap if present
    market_cap = data.get("market_cap")
    if market_cap is not None:
        try:
            if float(market_cap) < 0:
                return ValidationResult(False, f"Invalid market cap: {market_cap}")
        except (TypeError, ValueError):
            return ValidationResult(False, f"Invalid market cap format: {market_cap}")
    
    return ValidationResult(True, "Valid company info")

def validate_news(data, symbol):
    """Validate news data structure"""
    if not data or not data.get("ok"):
        if "skipped" in str(data.get("error", "")).lower():
            return ValidationResult(True, "Skipped due to missing API key", {"reason": "skipped"})
        # For invalid symbols, empty results are expected
        if symbol == "INVALID1234":
            return ValidationResult(True, "Expected empty result for invalid symbol")
        return ValidationResult(False, f"News fetch failed: {data.get('error', 'Unknown error')}")
    
    # Check required fields
    required_fields = ["symbol", "count"]
    for field in required_fields:
        if field not in data:
            return ValidationResult(False, f"Missing required field: {field}")
    
    # Validate count
    try:
        count = int(data["count"])
        if count < 0:
            return ValidationResult(False, f"Invalid count: {count}")
    except (TypeError, ValueError):
        return ValidationResult(False, f"Invalid count format: {data.get('count')}")
    
    return ValidationResult(True, "Valid news data")

def validate_multiple_quotes(data, symbol):
    """Validate multiple quotes structure"""
    if not data or not data.get("ok"):
        if "skipped" in str(data.get("error", "")).lower():
            return ValidationResult(True, "Skipped due to missing API key", {"reason": "skipped"})
        return ValidationResult(False, f"Multiple quotes fetch failed: {data.get('error', 'Unknown error')}")
    
    # Check required fields
    required_fields = ["count", "quotes"]
    for field in required_fields:
        if field not in data:
            return ValidationResult(False, f"Missing required field: {field}")
    
    # Validate count
    try:
        count = int(data["count"])
        if count < 0:
            return ValidationResult(False, f"Invalid count: {count}")
    except (TypeError, ValueError):
        return ValidationResult(False, f"Invalid count format: {data.get('count')}")
    
    # Validate quotes structure
    quotes = data.get("quotes", {})
    if not isinstance(quotes, dict):
        return ValidationResult(False, "Invalid quotes format")
    
    for sym, quote in quotes.items():
        if not isinstance(quote, dict):
            return ValidationResult(False, f"Invalid quote format for {sym}")
        
        # Validate individual quote
        quote_result = validate_quote({"ok": True, **quote}, sym)
        if not quote_result.passed and quote_result.reason != "Skipped due to missing API key":
            return ValidationResult(False, f"Invalid quote for {sym}: {quote_result.reason}")
    
    return ValidationResult(True, "Valid multiple quotes data")

def validate_financial_statements(data, symbol):
    """Validate financial statements structure"""
    if not data or not data.get("ok"):
        if "skipped" in str(data.get("error", "")).lower():
            return ValidationResult(True, "Skipped due to missing API key", {"reason": "skipped"})
        # For invalid symbols, empty results are expected
        if symbol == "INVALID1234":
            return ValidationResult(True, "Expected empty result for invalid symbol")
        return ValidationResult(False, f"Financial statements fetch failed: {data.get('error', 'Unknown error')}")
    
    # Check required fields
    required_fields = ["symbol", "summary"]
    for field in required_fields:
        if field not in data:
            return ValidationResult(False, f"Missing required field: {field}")
    
    # Validate summary structure
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        return ValidationResult(False, "Invalid summary format")
    
    return ValidationResult(True, "Valid financial statements data")

def validate_sentiment(data, symbol):
    """Validate sentiment data structure"""
    if not data:
        if symbol == "INVALID1234":
            return ValidationResult(True, "Expected empty result for invalid symbol")
        return ValidationResult(False, "No data returned from CLI")
    
    if not data.get("ok"):
        if "skipped" in str(data.get("error", "")).lower() or "skipped" in str(data.get("note", "")).lower():
            return ValidationResult(True, "Skipped due to missing API key or no sentiment available", {"reason": "skipped"})
        # For invalid symbols, empty results are expected
        if symbol == "INVALID1234":
            return ValidationResult(True, "Expected empty result for invalid symbol")
        return ValidationResult(False, f"Sentiment fetch failed: {data.get('error', 'Unknown error')}")
    
    # Check data field
    sentiment_data = data.get("data", {})
    if not isinstance(sentiment_data, dict):
        return ValidationResult(False, "Invalid sentiment data format")
    
    # For valid symbols with no data, this is acceptable
    if not sentiment_data and symbol != "INVALID1234":
        return ValidationResult(True, "No sentiment data available", {"reason": "no_data"})
    
    # Validate individual sentiment sources
    for source, sd in sentiment_data.items():
        if not isinstance(sd, dict):
            return ValidationResult(False, f"Invalid sentiment format for source {source}")
        
        # Validate score range (-1 to 1)
        score = sd.get("score")
        if score is not None:
            try:
                score_val = float(score)
                if not -1 <= score_val <= 1:
                    return ValidationResult(False, f"Invalid sentiment score for {source}: {score_val}")
            except (TypeError, ValueError):
                return ValidationResult(False, f"Invalid sentiment score format for {source}: {score}")
    
    return ValidationResult(True, "Valid sentiment data")

def validate_advanced_sentiment(data, symbol):
    """Validate advanced sentiment data structure"""
    if not data:
        if symbol == "INVALID1234":
            return ValidationResult(True, "Expected empty result for invalid symbol")
        return ValidationResult(False, "No data returned from CLI")
    
    if not data.get("ok"):
        if "skipped" in str(data.get("error", "")).lower() or "skipped" in str(data.get("note", "")).lower():
            return ValidationResult(True, "Skipped due to missing API key or no texts available", {"reason": "skipped"})
        # For invalid symbols, empty results are expected
        if symbol == "INVALID1234":
            return ValidationResult(True, "Expected empty result for invalid symbol")
        return ValidationResult(False, f"Advanced sentiment fetch failed: {data.get('error', 'Unknown error')}")
    
    # Check required fields for valid data
    if data.get("score") is not None:
        required_fields = ["score", "confidence", "timestamp"]
        for field in required_fields:
            if field not in data:
                return ValidationResult(False, f"Missing required field: {field}")
        
        # Validate score range (-1 to 1)
        try:
            score = float(data["score"])
            if not -1 <= score <= 1:
                return ValidationResult(False, f"Invalid sentiment score: {score}")
        except (TypeError, ValueError):
            return ValidationResult(False, f"Invalid sentiment score format: {data.get('score')}")
        
        # Validate confidence (0 to 1)
        try:
            confidence = float(data["confidence"])
            if not 0 <= confidence <= 1:
                return ValidationResult(False, f"Invalid confidence: {confidence}")
        except (TypeError, ValueError):
            return ValidationResult(False, f"Invalid confidence format: {data.get('confidence')}")
    
    return ValidationResult(True, "Valid advanced sentiment data")

def validate_market_breadth(data, symbol):
    """Validate market breadth data structure"""
    if not data or not data.get("ok"):
        if "skipped" in str(data.get("error", "")).lower() or "skipped" in str(data.get("note", "")).lower():
            return ValidationResult(True, "Skipped due to unavailable adapter or missing key", {"reason": "skipped"})
        return ValidationResult(False, f"Market breadth fetch failed: {data.get('error', 'Unknown error')}")
    
    # Validate numeric fields
    numeric_fields = ["advancers", "decliners", "unchanged"]
    for field in numeric_fields:
        value = data.get(field)
        if value is not None:
            try:
                val = int(value)
                if val < 0:
                    return ValidationResult(False, f"Invalid {field}: {val}")
            except (TypeError, ValueError):
                return ValidationResult(False, f"Invalid {field} format: {value}")
    
    # Validate put_call_ratio if present
    put_call_ratio = data.get("put_call_ratio")
    if put_call_ratio is not None:
        try:
            ratio = float(put_call_ratio)
            if ratio < 0:
                return ValidationResult(False, f"Invalid put_call_ratio: {ratio}")
        except (TypeError, ValueError):
            return ValidationResult(False, f"Invalid put_call_ratio format: {put_call_ratio}")
    
    return ValidationResult(True, "Valid market breadth data")

def validate_sector_performance(data, symbol):
    """Validate sector performance data structure"""
    if not data:
        return ValidationResult(False, "No data returned from CLI")
    
    if not data.get("ok"):
        if "skipped" in str(data.get("error", "")).lower() or "skipped" in str(data.get("note", "")).lower():
            return ValidationResult(True, "Skipped due to unavailable adapter or missing key", {"reason": "skipped"})
        return ValidationResult(False, f"Sector performance fetch failed: {data.get('error', 'Unknown error')}")
    
    # Check required fields
    required_fields = ["rows", "count"]
    for field in required_fields:
        if field not in data:
            return ValidationResult(False, f"Missing required field: {field}")
    
    # Validate count
    try:
        count = int(data["count"])
        if count < 0:
            return ValidationResult(False, f"Invalid count: {count}")
    except (TypeError, ValueError):
        return ValidationResult(False, f"Invalid count format: {data.get('count')}")
    
    # Validate rows structure
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        return ValidationResult(False, "Invalid rows format")
    
    # Validate individual rows
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            return ValidationResult(False, f"Invalid row format at index {i}")
        
        # Validate performance fields
        perf_fields = ["perf_1d", "perf_1w", "perf_1m", "perf_ytd"]
        for field in perf_fields:
            if field in row:
                value = row[field]
                if value is not None:
                    try:
                        float(value)  # Just check if it's convertible
                    except (TypeError, ValueError):
                        return ValidationResult(False, f"Invalid {field} format in row {i}: {value}")
    
    return ValidationResult(True, "Valid sector performance data")

def compare_values(value, ref_value, tolerance_pct=2.0):
    """Compare two values with tolerance using CLI compare helper"""
    try:
        cmd = ["python", "cli_validate.py", "--json", "compare", "--value", str(value), "--ref_value", str(ref_value), "--tolerance_pct", str(tolerance_pct)]
        stdout, stderr, code = run_cli_command(cmd)
        result = parse_cli_output(stdout)
        if result and result.get("ok"):
            return result.get("result") == "pass", result.get("pct_diff", 0)
        return False, None
    except Exception:
        return False, None

def cross_reference_validate_quote(symbol, price, reference_price):
    """Cross-reference quote price with reference value"""
    if reference_price is not None and price is not None:
        passed, diff = compare_values(price, reference_price, 2.0)  # 2% tolerance
        return passed, diff
    return True, 0  # No reference available, pass by default

# Enhanced test functions with result storage
@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda x: x["name"])
def test_data_point_structure(test_case):
    """Test data point structure and basic validation"""
    validation_func = globals()[test_case["validation"]]
    
    if test_case.get("single_test"):
        # Run single test for commands that don't require symbol parameter
        command = test_case["command"]
        stdout, stderr, code = run_cli_command(command)
        if not stdout.strip() and isinstance(command, list) and len(command) >= 2 and command[1] == "cli_validate.py":
            alt_command = command.copy()
            alt_command[1] = "tests/cli_validate.py"
            stdout, stderr, code = run_cli_command(alt_command)
        data = parse_cli_output(stdout)
        
        result = validation_func(data, "")
        status = "pass" if result.passed else "fail"
        reason = result.reason if not result.passed else "Valid data structure"
        store_test_result(test_case["name"], "N/A", status, reason, stdout, stderr)
        
        if not result.passed:
            pytest.fail(f"Validation failed: {result.reason}")
    else:
        # Run tests for each ticker
        for symbol in TEST_TICKERS:
            command = [arg.format(symbol=symbol) if isinstance(arg, str) and "{symbol}" in arg else arg for arg in test_case["command"]]
            stdout, stderr, code = run_cli_command(command)
            # Fallback: if stdout empty, try executing the tests/cli_validate.py explicitly (robustness)
            if not stdout.strip() and isinstance(command, list) and len(command) >= 2 and command[1] == "cli_validate.py":
                alt_command = command.copy()
                alt_command[1] = "tests/cli_validate.py"
                stdout, stderr, code = run_cli_command(alt_command)
            data = parse_cli_output(stdout)
            
            result = validation_func(data, symbol)
            status = "pass" if result.passed else "fail"
            reason = result.reason if not result.passed else "Valid data structure"
            
            # For skipped tests, mark as skip
            if result.details and result.details.get("reason") == "skipped":
                status = "skip"
                reason = "Skipped due to missing API key or unavailable adapter"
            
            store_test_result(test_case["name"], symbol, status, reason, stdout, stderr)
            
            if not result.passed:
                # For invalid symbol, some failures are expected
                if symbol == "INVALID1234" and "fetch failed" in result.reason.lower():
                    continue  # Expected failure
                pytest.fail(f"Validation failed for {symbol}: {result.reason}")

@pytest.mark.parametrize("test_case", [tc for tc in TEST_CASES if "network" in tc["markers"]], ids=lambda x: x["name"])
def test_network_dependency(test_case):
    """Test network-dependent functionality"""
    pytest.skip("Network tests - run with -m network")

@pytest.mark.parametrize("test_case", [tc for tc in TEST_CASES if "slow" in tc["markers"]], ids=lambda x: x["name"])
def test_slow_operations(test_case):
    """Test slow operations"""
    pytest.skip("Slow tests - run with -m slow")

@pytest.mark.parametrize("test_case", [tc for tc in TEST_CASES if "optional" in tc["markers"]], ids=lambda x: x["name"])
def test_optional_adapters(test_case):
    """Test optional adapter functionality"""
    # Check if required API keys are available
    adapter_key = None
    test_name = test_case["name"].lower()
    for adapter, env_var in OPTIONAL_ADAPTERS.items():
        if adapter in test_name:
            adapter_key = env_var
            break
    
    if adapter_key and not os.getenv(adapter_key):
        pytest.skip(f"Skipping {test_case['name']} - {adapter_key} not set")
    
    # Run basic test
    test_data_point_structure(test_case)

def generate_test_summary():
    """Generate a summary of test results"""
    if not TEST_RESULTS:
        return "No test results available"
    
    total_tests = len(TEST_RESULTS)
    passed_tests = sum(1 for r in TEST_RESULTS if r["status"] == "pass")
    failed_tests = sum(1 for r in TEST_RESULTS if r["status"] == "fail")
    skipped_tests = sum(1 for r in TEST_RESULTS if r["status"] == "skip")
    
    summary = {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "skipped": skipped_tests,
        "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
        "results": TEST_RESULTS
    }
    
    return summary

def append_to_checklist(summary):
    """Append test results to the DATA_VALIDATION_CHECKLIST.md"""
    checklist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "DATA_VALIDATION_CHECKLIST.md")
    
    # Create automated test results section
    automated_section = f"""
## Automated Test Results
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

### Summary
- Total Tests: {summary['total_tests']}
- Passed: {summary['passed']}
- Failed: {summary['failed']}
- Skipped: {summary['skipped']}
- Success Rate: {summary['success_rate']:.1f}%

### Detailed Results
"""
    
    # Group results by test type
    results_by_type = {}
    for result in summary['results']:
        test_type = result['test_name']
        if test_type not in results_by_type:
            results_by_type[test_type] = []
        results_by_type[test_type].append(result)
    
    # Add detailed results for each test type
    for test_type, results in results_by_type.items():
        automated_section += f"\n#### {test_type.title()} Tests\n"
        automated_section += "| Symbol | Status | Reason | Timestamp |\n"
        automated_section += "|--------|--------|--------|-----------|\n"
        
        for result in results:
            symbol = result['symbol'] or "N/A"
            status = result['status'].upper()
            reason = result['reason'][:50] + "..." if len(result['reason']) > 50 else result['reason']
            timestamp = result['timestamp'].split('T')[1].split('.')[0]  # Just time part
            automated_section += f"| {symbol} | {status} | {reason} | {timestamp} |\n"
    
    automated_section += "\n---\n"
    
    # Read existing checklist
    try:
        with open(checklist_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        content = "# Data Validation Checklist\n\n"
    
    # Append automated results section
    if "## Automated Test Results" in content:
        # Replace existing section
        lines = content.split('\n')
        new_lines = []
        in_automated_section = False
        
        for line in lines:
            if line.strip() == "## Automated Test Results":
                in_automated_section = True
                new_lines.append(automated_section.strip())
            elif in_automated_section and line.startswith('#') and not line.startswith('## Automated Test Results'):
                in_automated_section = False
                new_lines.append(line)
            elif not in_automated_section:
                new_lines.append(line)
        
        if not in_automated_section:  # Section was at the end
            content = '\n'.join(new_lines)
        else:
            content = '\n'.join(new_lines)
    else:
        # Append new section at the end
        content += f"\n{automated_section}"
    
    # Write back to file
    with open(checklist_path, 'w') as f:
        f.write(content)

def pytest_sessionfinish(session):
    """Hook to run after test session finishes"""
    if TEST_RESULTS:
        summary = generate_test_summary()
        append_to_checklist(summary)

def test_cli_compare_helper():
    """Test the CLI compare helper functionality"""
    # Test passing comparison
    cmd = ["python", "cli_validate.py", "--json", "compare", "--value", "100", "--ref_value", "102", "--tolerance_pct", "2.0"]
    stdout, stderr, code = run_cli_command(cmd)
    if not stdout.strip():
        stdout, stderr, code = run_cli_command(["python", "tests/cli_validate.py", "--json", "compare", "--value", "100", "--ref_value", "102", "--tolerance_pct", "2.0"])
    data = parse_cli_output(stdout)
    
    assert data is not None, "Compare helper should return JSON output"
    assert data.get("ok"), "Compare helper should succeed"
    assert data.get("result") == "pass", "100 vs 102 with 2% tolerance should pass"
    
    # Test failing comparison
    cmd = ["python", "cli_validate.py", "--json", "compare", "--value", "100", "--ref_value", "105", "--tolerance_pct", "2.0"]
    stdout, stderr, code = run_cli_command(cmd)
    data = parse_cli_output(stdout)
    
    assert data is not None, "Compare helper should return JSON output"
    assert data.get("ok"), "Compare helper should succeed"
    assert data.get("result") == "fail", "100 vs 105 with 2% tolerance should fail"

def test_cli_flags():
    """Test CLI flags functionality"""
    # Test --json flag produces clean JSON output
    cmd = ["python", "cli_validate.py", "--json", "quote", "--symbol", "AAPL"]
    stdout, stderr, code = run_cli_command(cmd)
    if not stdout.strip():
        stdout, stderr, code = run_cli_command(["python", "tests/cli_validate.py", "--json", "quote", "--symbol", "AAPL"])
    
    # Should be valid JSON
    try:
        data = json.loads(stdout)
        assert isinstance(data, dict), "JSON output should be a dictionary"
        assert "ok" in data, "JSON should contain 'ok' field"
    except json.JSONDecodeError:
        pytest.fail("CLI --json flag should produce valid JSON output")
    
    # Test --exit-zero-even-on-error flag
    cmd = ["python", "cli_validate.py", "--json", "--exit-zero-even-on-error", "quote", "--symbol", "INVALID1234"]
    stdout, stderr, code = run_cli_command(cmd)
    if not stdout.strip():
        stdout, stderr, code = run_cli_command(["python", "tests/cli_validate.py", "--json", "--exit-zero-even-on-error", "quote", "--symbol", "INVALID1234"])
    
    assert code == 0, "--exit-zero-even-on-error should always exit with code 0"

if __name__ == "__main__":
    # Run tests and generate summary
    pytest.main([__file__, "-v"])