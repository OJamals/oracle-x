#!/usr/bin/env python3
"""
Test Oracle Pipeline Integration for Dashboard
"""
import os
import sys
import json
from datetime import datetime

# Add parent dir to sys.path for backend import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_pipeline_import():
    """Test if the pipeline can be imported correctly"""
    print("=== Testing Pipeline Import ===")

    try:
        # Simulate the dashboard's import logic
        run_oracle_pipeline = None
        try:
            if os.environ.get("STREAMLIT") or __name__ == "__main__":
                from main import run_oracle_pipeline  # type: ignore
                print("✅ Pipeline imported successfully")
                return run_oracle_pipeline
            else:
                print("⚠️  STREAMLIT environment not set, pipeline not imported")
                return None
        except Exception as e:
            print(f"❌ Pipeline import failed: {e}")
            return None
    except Exception as e:
        print(f"❌ Pipeline import test failed: {e}")
        return None

def test_pipeline_execution():
    """Test pipeline execution with a simple prompt"""
    print("\n=== Testing Pipeline Execution ===")

    # Set STREAMLIT env to enable import
    os.environ["STREAMLIT"] = "1"

    try:
        run_oracle_pipeline = test_pipeline_import()
        if not run_oracle_pipeline:
            print("❌ Cannot test execution without pipeline import")
            return False

        if not callable(run_oracle_pipeline):
            print("❌ Pipeline runner is not callable")
            return False

        # Test with a simple prompt
        test_prompt = "Test market summary for TSLA and SPY"
        print(f"Running pipeline with prompt: {test_prompt}")

        # This might take a while, so we'll wrap it in a try-catch
        try:
            results = run_oracle_pipeline(test_prompt)
            print(f"✅ Pipeline executed successfully")
            print(f"Result keys: {list(results.keys())}")
            print(f"Date: {results.get('date', 'N/A')}")

            if 'playbook' in results:
                playbook = results['playbook']
                if isinstance(playbook, dict) and 'trades' in playbook:
                    trades = playbook['trades']
                    print(f"Found {len(trades)} trades in playbook")
                elif isinstance(playbook, list):
                    print(f"Playbook is a list with {len(playbook)} items")

            return True

        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Pipeline execution test failed: {e}")
        return False
    finally:
        # Clean up environment
        if "STREAMLIT" in os.environ:
            del os.environ["STREAMLIT"]

def test_pipeline_structure():
    """Test the expected pipeline output structure"""
    print("\n=== Testing Pipeline Output Structure ===")

    try:
        # Look at recent playbook files to understand expected structure
        playbooks_dir = "playbooks/"
        if not os.path.exists(playbooks_dir):
            print("❌ Playbooks directory not found")
            return False

        json_files = [f for f in os.listdir(playbooks_dir) if f.endswith('.json')]
        if not json_files:
            print("❌ No playbook JSON files found")
            return False

        latest_file = max(json_files, key=lambda x: os.path.getmtime(os.path.join(playbooks_dir, x)))
        playbook_path = os.path.join(playbooks_dir, latest_file)

        with open(playbook_path, 'r') as f:
            playbook_data = json.load(f)

        print(f"Analyzing structure of: {latest_file}")

        # Check for expected structure
        if "playbook" in playbook_data:
            inner_playbook = playbook_data["playbook"]
            if "trades" in inner_playbook:
                trades = inner_playbook["trades"]
                print(f"✅ Nested structure found: {len(trades)} trades")
                if trades:
                    sample_trade = trades[0]
                    required_fields = ["ticker", "direction", "instrument", "entry_range", "profit_target", "stop_loss", "thesis"]
                    missing_fields = [field for field in required_fields if field not in sample_trade]
                    if missing_fields:
                        print(f"⚠️  Missing fields in trade: {missing_fields}")
                    else:
                        print("✅ All required trade fields present")
            if "tomorrows_tape" in inner_playbook:
                print("✅ Tomorrow's tape found")
        elif "trades" in playbook_data:
            trades = playbook_data["trades"]
            print(f"✅ Direct structure found: {len(trades)} trades")
        else:
            print("⚠️  Unexpected playbook structure")

        return True

    except Exception as e:
        print(f"❌ Pipeline structure test failed: {e}")
        return False

def main():
    """Run all pipeline integration tests"""
    print("🔧 Dashboard Pipeline Integration Tests")
    print("=" * 50)

    tests_passed = 0
    total_tests = 3

    # Test 1: Pipeline import
    pipeline_runner = test_pipeline_import()
    if pipeline_runner is not None:
        tests_passed += 1

    # Test 2: Pipeline execution (only if import worked)
    if pipeline_runner is not None:
        if test_pipeline_execution():
            tests_passed += 1
    else:
        print("\n⚠️  Skipping pipeline execution test due to import failure")

    # Test 3: Pipeline structure
    if test_pipeline_structure():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"Pipeline integration tests completed: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All pipeline integration tests passed!")
    else:
        print("⚠️  Some pipeline integration tests failed - check output above")

    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)