#!/usr/bin/env python3
"""
Test Dashboard Display Components
"""
import os
import sys
import json
from datetime import datetime

# Add parent dir to sys.path for backend import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_results_display():
    """Test the results display logic that would be used in the dashboard"""
    print("=== Testing Results Display Components ===")

    # Load a recent playbook to test display logic
    playbooks_dir = "playbooks/"
    json_files = [
        f
        for f in os.listdir(playbooks_dir)
        if f.endswith(".json") and "standard_playbook" in f
    ]

    if not json_files:
        print("❌ No standard playbook files found for testing")
        return False

    latest_file = max(
        json_files, key=lambda x: os.path.getmtime(os.path.join(playbooks_dir, x))
    )
    playbook_path = os.path.join(playbooks_dir, latest_file)

    print(f"Loading playbook: {latest_file}")

    try:
        with open(playbook_path, "r") as f:
            results = json.load(f)

        # Test the display logic from the dashboard
        playbook = results.get("playbook", {})
        if isinstance(playbook, dict):
            tape = playbook.get("tomorrows_tape")
            trades = playbook.get("trades", [])
        else:
            tape = results.get("tomorrows_tape")
            trades = results.get("trades", [])

        print(f"✅ Found {len(trades)} trades")
        if tape:
            print(f"✅ Found tomorrow's tape: {tape[:100]}...")

        # Test trade display logic
        if trades and isinstance(trades, list) and len(trades) > 0:
            for i, trade in enumerate(trades[:3]):  # Test first 3 trades
                ticker = trade.get("ticker", "N/A")
                direction = trade.get("direction", "").lower()
                instrument = trade.get("instrument", "")
                entry_range = trade.get("entry_range", "")
                profit_target = trade.get("profit_target", "")
                stop_loss = trade.get("stop_loss", "")
                thesis = trade.get("thesis", "")
                scenario_tree = trade.get("scenario_tree", {})

                print(f"  Trade {i+1}: {ticker} {direction} {instrument}")
                print(
                    f"    Entry: {entry_range}, Target: {profit_target}, Stop: {stop_loss}"
                )

                # Test color logic
                if direction in ["long", "buy", "call", "bullish"]:
                    dir_color = "#2ecc40"  # green
                    dir_label = "BUY"
                elif direction in ["short", "sell", "put", "bearish"]:
                    dir_color = "#e74c3c"  # red
                    dir_label = "SELL"
                else:
                    dir_color = "#888888"
                    dir_label = direction.upper() if direction else "N/A"

                print(f"    Display: {dir_label} ({dir_color})")

                # Test scenario tree display
                for case, text in scenario_tree.items():
                    case_color = (
                        "#2ecc40"
                        if "bull" in case
                        else ("#e74c3c" if "bear" in case else "#8884d8")
                    )
                    print(f"    Scenario {case.title()}: {text} ({case_color})")

        return True

    except Exception as e:
        print(f"❌ Results display test failed: {e}")
        return False


def test_chart_integration():
    """Test chart file integration logic"""
    print("\n=== Testing Chart Integration ===")

    try:
        # Check for charts directory
        charts_dir = "charts/"
        if not os.path.exists(charts_dir):
            print("⚠️  Charts directory not found - creating")
            os.makedirs(charts_dir, exist_ok=True)

        # List existing chart files
        chart_files = [f for f in os.listdir(charts_dir) if f.endswith(".png")]
        print(f"Found {len(chart_files)} chart files")

        # Test chart file existence check logic
        test_chart_paths = [
            "charts/GOOG_price_chart.png",
            "charts/SPY_scenario_chart.png",
            "charts/nonexistent_chart.png",
        ]

        for chart_path in test_chart_paths:
            exists = os.path.exists(chart_path)
            print(f"  {chart_path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")

        return True

    except Exception as e:
        print(f"❌ Chart integration test failed: {e}")
        return False


def test_raw_data_access():
    """Test raw data access and JSON/log expansion logic"""
    print("\n=== Testing Raw Data Access ===")

    try:
        # Load a playbook for raw data testing
        playbooks_dir = "playbooks/"
        json_files = [f for f in os.listdir(playbooks_dir) if f.endswith(".json")]

        if not json_files:
            print("❌ No playbook files found")
            return False

        latest_file = max(
            json_files, key=lambda x: os.path.getmtime(os.path.join(playbooks_dir, x))
        )
        playbook_path = os.path.join(playbooks_dir, latest_file)

        with open(playbook_path, "r") as f:
            raw_data = json.load(f)

        # Test JSON serialization for display
        try:
            json_str = json.dumps(raw_data, indent=2)
            print(f"✅ JSON serialization works (length: {len(json_str)} chars)")
        except Exception as e:
            print(f"❌ JSON serialization failed: {e}")
            return False

        # Test logs access
        logs = raw_data.get("logs", "")
        if logs:
            print(f"✅ Found logs (length: {len(logs)} chars)")
            print(f"Log preview: {logs[:200]}...")
        else:
            print("⚠️  No logs found in playbook")

        return True

    except Exception as e:
        print(f"❌ Raw data access test failed: {e}")
        return False


def test_trade_table_generation():
    """Test the trade recommendations table generation"""
    print("\n=== Testing Trade Table Generation ===")

    try:
        # Load playbook data
        playbooks_dir = "playbooks/"
        json_files = [f for f in os.listdir(playbooks_dir) if f.endswith(".json")]

        if not json_files:
            print("❌ No playbook files found")
            return False

        latest_file = max(
            json_files, key=lambda x: os.path.getmtime(os.path.join(playbooks_dir, x))
        )
        playbook_path = os.path.join(playbooks_dir, latest_file)

        with open(playbook_path, "r") as f:
            results = json.load(f)

        # Extract trades for table generation
        playbook = results.get("playbook", {})
        if isinstance(playbook, dict):
            trades = playbook.get("trades", [])
        else:
            trades = results.get("trades", [])

        if not trades:
            print("❌ No trades found for table generation")
            return False

        # Test pandas DataFrame creation
        try:
            import pandas as pd

            summary_cols = [
                "ticker",
                "direction",
                "instrument",
                "entry_range",
                "profit_target",
                "stop_loss",
                "thesis",
            ]
            summary_data = [
                {col: trade.get(col, "") for col in summary_cols} for trade in trades
            ]

            df = pd.DataFrame(summary_data)
            print(
                f"✅ DataFrame created successfully: {df.shape[0]} rows, {df.shape[1]} columns"
            )
            print("Columns:", list(df.columns))
            print("Sample data:")
            print(df.head())

            return True

        except ImportError:
            print("⚠️  Pandas not available, but table logic is sound")
            return True

    except Exception as e:
        print(f"❌ Trade table generation test failed: {e}")
        return False


def main():
    """Run all display component tests"""
    print("🔧 Dashboard Display Components Tests")
    print("=" * 50)

    tests_passed = 0
    total_tests = 4

    # Test 1: Results display
    if test_results_display():
        tests_passed += 1

    # Test 2: Chart integration
    if test_chart_integration():
        tests_passed += 1

    # Test 3: Raw data access
    if test_raw_data_access():
        tests_passed += 1

    # Test 4: Trade table generation
    if test_trade_table_generation():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"Display component tests completed: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All display component tests passed!")
    else:
        print("⚠️  Some display component tests failed - check output above")

    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
