#!/usr/bin/env python3
"""
Integration test for Dashboard with Oracle Pipeline
"""
import os
import sys
import json

# Add parent dir to sys.path for backend import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_dashboard_auto_generate():
    """Test the dashboard's auto-generate functionality"""
    print("=== Testing Dashboard Auto-Generate Integration ===")

    # Import dashboard functions
    from app import auto_generate_market_summary, list_playbooks, load_playbook

    try:
        # Test playbook listing
        playbooks = list_playbooks()
        print(f"Found {len(playbooks)} playbooks")

        if playbooks:
            # Test loading the latest playbook
            latest = playbooks[0]
            print(f"Testing with latest playbook: {latest}")

            playbook_data = load_playbook(latest)
            print(f"Playbook keys: {list(playbook_data.keys())}")

            # Test auto-generate function
            summary = auto_generate_market_summary()
            print(f"Generated summary: {summary}")

            # Verify summary contains expected elements
            if "Tomorrow's tape:" in summary and "Key trades:" in summary:
                print("✅ Summary format is correct")
                return True
            else:
                print("⚠️  Summary format may be incorrect")
                return False

        return False

    except Exception as e:
        print(f"❌ Dashboard integration test failed: {e}")
        return False


def test_pipeline_results_parsing():
    """Test parsing pipeline results in dashboard format"""
    print("\n=== Testing Pipeline Results Parsing ===")

    try:
        # Simulate pipeline results format
        pipeline_results = {
            "date": "2025-08-21",
            "playbook": '{"trades": [{"ticker": "GOOG", "direction": "long", "instrument": "shares", "entry_range": "190-193", "profit_target": "200", "stop_loss": "185", "thesis": "Strong institutional support", "scenario_tree": {"base_case": "50% - Consolidates", "bull_case": "30% - Rally", "bear_case": "20% - Weakness"}}], "tomorrows_tape": "Markets are poised for a mixed session"}',
            "logs": "Pipeline completed successfully",
        }

        # Test parsing the playbook JSON string
        playbook_str = pipeline_results["playbook"]
        if isinstance(playbook_str, str):
            try:
                playbook_data = json.loads(playbook_str)
                trades = playbook_data.get("trades", [])
                tape = playbook_data.get("tomorrows_tape")

                print(f"✅ Successfully parsed {len(trades)} trades from JSON string")
                print(f"✅ Found tomorrow's tape: {tape[:50]}...")

                if trades:
                    sample_trade = trades[0]
                    print(
                        f"✅ Sample trade: {sample_trade.get('ticker')} {sample_trade.get('direction')}"
                    )

                return True

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse playbook JSON string: {e}")
                return False
        else:
            print("❌ Playbook is not a string format")
            return False

    except Exception as e:
        print(f"❌ Pipeline results parsing test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("🔧 Dashboard Integration Tests")
    print("=" * 50)

    tests_passed = 0
    total_tests = 2

    # Test 1: Dashboard auto-generate
    if test_dashboard_auto_generate():
        tests_passed += 1

    # Test 2: Pipeline results parsing
    if test_pipeline_results_parsing():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"Integration tests completed: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All integration tests passed!")
    else:
        print("⚠️  Some integration tests failed - check output above")

    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
