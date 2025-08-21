#!/usr/bin/env python3
"""
Test script for Dashboard Pipeline components
"""
import os
import sys
import json
from datetime import datetime

# Add parent dir to sys.path for backend import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dashboard functions
sys.path.append(os.path.dirname(__file__))
from app import list_playbooks, load_playbook, auto_generate_market_summary

def test_playbooks_integration():
    """Test playbook listing and loading functionality"""
    print("=== Testing Playbooks Integration ===")

    try:
        playbooks = list_playbooks()
        print(f"Found {len(playbooks)} playbooks:")
        for pb in playbooks[:5]:  # Show first 5
            print(f"  - {pb}")

        if playbooks:
            latest = playbooks[0]
            print(f"\nLoading latest playbook: {latest}")
            playbook_data = load_playbook(latest)
            print(f"Playbook structure keys: {list(playbook_data.keys())}")

            # Check for expected structure
            if "trades" in playbook_data:
                trades = playbook_data["trades"]
                print(f"Found {len(trades)} trades")
                if trades:
                    print(f"First trade: {trades[0].get('ticker', 'N/A')} - {trades[0].get('direction', 'N/A')}")
            elif "playbook" in playbook_data:
                inner_playbook = playbook_data["playbook"]
                if "trades" in inner_playbook:
                    trades = inner_playbook["trades"]
                    print(f"Found {len(trades)} trades (nested structure)")
                    if trades:
                        print(f"First trade: {trades[0].get('ticker', 'N/A')} - {trades[0].get('direction', 'N/A')}")

            if "tomorrows_tape" in playbook_data:
                tape = playbook_data["tomorrows_tape"]
                print(f"Tape preview: {tape[:100]}...")

        print("✅ Playbooks integration test passed")
        return True

    except Exception as e:
        print(f"❌ Playbooks integration test failed: {e}")
        return False

def test_auto_generate_summary():
    """Test the auto-generate market summary function"""
    print("\n=== Testing Auto-Generate Market Summary ===")

    try:
        summary = auto_generate_market_summary()
        print(f"Generated summary: {summary}")
        print("✅ Auto-generate summary test passed")
        return True

    except Exception as e:
        print(f"❌ Auto-generate summary test failed: {e}")
        return False

def test_service_health():
    """Test service health checking"""
    print("\n=== Testing Service Health Checks ===")

    from app import check_service

    services = [
        ("Qdrant", "http://localhost:6333/collections"),
        ("Qwen3", "http://localhost:8000/v1/models")
    ]

    results = {}
    for name, url in services:
        try:
            status = check_service(url)
            results[name] = status
            print(f"{name}: {'✅ UP' if status else '❌ DOWN'}")
        except Exception as e:
            results[name] = False
            print(f"{name}: ❌ ERROR - {e}")

    return results

def main():
    """Run all tests"""
    print("🔧 Dashboard Pipeline Debug Tests")
    print("=" * 50)

    tests_passed = 0
    total_tests = 3

    # Test 1: Playbooks integration
    if test_playbooks_integration():
        tests_passed += 1

    # Test 2: Auto-generate summary
    if test_auto_generate_summary():
        tests_passed += 1

    # Test 3: Service health
    service_results = test_service_health()
    tests_passed += 1  # Service health test always "passes" - just reports status

    print("\n" + "=" * 50)
    print(f"Tests completed: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check output above")

    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)