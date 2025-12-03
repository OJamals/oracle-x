#!/usr/bin/env python3
"""
🧪 ORACLE-X Unified Test Runner

A comprehensive test management system for the ORACLE-X trading intelligence platform.
Provides organized test execution, reporting, and validation capabilities.

Usage Examples:
    python test_runner.py --all                    # Run all tests
    python test_runner.py --unit                   # Run unit tests only
    python test_runner.py --integration            # Run integration tests only
    python test_runner.py --performance            # Run performance tests
    python test_runner.py --validate-system       # Validate system health
    python test_runner.py --fast                   # Run critical tests only
    python test_runner.py --report                 # Generate test report
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class TestRunner:
    """Unified test runner for ORACLE-X system"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {}
        self.start_time = time.time()

    def run_pytest(
        self, test_path: str, markers: Optional[str] = None, verbose: bool = True
    ) -> Dict[str, Any]:
        """Run pytest with specified parameters"""
        cmd = ["python", "-m", "pytest", test_path]

        if markers:
            cmd.extend(["-m", markers])

        if verbose:
            cmd.append("-v")

        # Add coverage and reporting
        cmd.extend(["--tb=short", "--durations=10"])

        print(f"🚀 Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )
            return {
                "command": " ".join(cmd),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }
        except Exception as e:
            return {
                "command": " ".join(cmd),
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
            }

    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        print("🧪 Running Unit Tests...")
        return self.run_pytest("tests/unit/", verbose=True)

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        print("🔗 Running Integration Tests...")
        return self.run_pytest("tests/integration/", verbose=True)

    def run_critical_tests(self) -> List[Dict[str, Any]]:
        """Run critical system tests"""
        print("⚡ Running Critical Tests...")
        critical_tests = [
            "tests/test_financial_calculator.py",
            "tests/test_enhanced_sentiment_pipeline.py",
            "tests/unit/test_basic_system.py",
        ]

        results = []
        for test in critical_tests:
            if os.path.exists(test):
                result = self.run_pytest(test, verbose=False)
                results.append(result)
            else:
                print(f"⚠️  Critical test not found: {test}")

        return results

    def validate_system_health(self) -> Dict[str, Any]:
        """Validate system imports and basic functionality"""
        print("🏥 Validating System Health...")

        validation_script = """
import sys
import traceback

tests = {
    "Main Pipeline": "import main",
    "Options Pipeline": "import oracle_options_pipeline",
    "Data Orchestrator": "from data_feeds.data_feed_orchestrator import DataFeedOrchestrator",
    "Environment Config": "from core.config import config, load_config; load_config()",
    "ML Model Manager": "from oracle_engine.ml_model_manager import MLModelManager",
    "Prompt Chain": "from oracle_engine.chains.prompt_chain import adjust_scenario_tree"
}

results = {}
for name, test_code in tests.items():
    try:
        exec(test_code)
        results[name] = {"status": "✅ PASS", "error": None}
    except Exception as e:
        results[name] = {"status": "❌ FAIL", "error": str(e)}

print("System Health Validation Results:")
print("=" * 50)
for name, result in results.items():
    print(f"{result['status']} {name}")
    if result['error']:
        print(f"    Error: {result['error']}")

all_passed = all(r["status"] == "✅ PASS" for r in results.values())
exit_code = 0 if all_passed else 1
sys.exit(exit_code)
"""

        try:
            result = subprocess.run(
                [sys.executable, "-c", validation_script],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return {
                "command": "System Health Validation",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }
        except Exception as e:
            return {
                "command": "System Health Validation",
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
            }

    def run_performance_tests(self) -> List[Dict[str, Any]]:
        """Run performance-focused tests"""
        print("⚡ Running Performance Tests...")

        performance_tests = [
            "tests/test_data_feed_orchestrator_basic.py",
            "tests/test_ml_engine_simple.py",
        ]

        results = []
        for test in performance_tests:
            if os.path.exists(test):
                result = self.run_pytest(test, verbose=False)
                results.append(result)

        return results

    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        duration = time.time() - self.start_time

        report = f"""
🧪 ORACLE-X Test Execution Report
{'=' * 60}
⏱️  Total Duration: {duration:.2f} seconds
📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

📊 Test Results Summary:
"""

        total_tests = 0
        passed_tests = 0

        for test_name, result in self.test_results.items():
            if isinstance(result, list):
                # Handle list of results
                list_passed = sum(1 for r in result if r.get("success", False))
                list_total = len(result)
                status = f"{list_passed}/{list_total} passed"

                total_tests += list_total
                passed_tests += list_passed
            else:
                # Handle single result
                status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                total_tests += 1
                if result.get("success", False):
                    passed_tests += 1

            report += f"  {status:<15} {test_name}\n"

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        report += f"""
🎯 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})

📝 Detailed Results:
"""

        for test_name, result in self.test_results.items():
            if isinstance(result, list):
                report += f"\n{test_name}:\n"
                for i, r in enumerate(result, 1):
                    status = "✅" if r.get("success", False) else "❌"
                    report += f"  {status} Test {i}: {r.get('command', 'Unknown')}\n"
            else:
                status = "✅" if result.get("success", False) else "❌"
                report += f"\n{status} {test_name}:\n"
                if not result.get("success", False) and result.get("stderr"):
                    report += f"  Error: {result['stderr'][:200]}...\n"

        return report

    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Running Complete Test Suite...")

        # System health first
        self.test_results["System Health"] = self.validate_system_health()

        # Unit tests
        if os.path.exists("tests/unit"):
            self.test_results["Unit Tests"] = self.run_unit_tests()

        # Integration tests
        if os.path.exists("tests/integration"):
            self.test_results["Integration Tests"] = self.run_integration_tests()

        # Critical tests
        self.test_results["Critical Tests"] = self.run_critical_tests()

    def run_fast_tests(self):
        """Run essential tests quickly"""
        print("⚡ Running Fast Test Suite...")

        # System health and critical tests only
        self.test_results["System Health"] = self.validate_system_health()
        self.test_results["Critical Tests"] = self.run_critical_tests()


def main():
    parser = argparse.ArgumentParser(
        description="🧪 ORACLE-X Unified Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py --all                 # Run all tests
  python test_runner.py --fast                # Run critical tests only
  python test_runner.py --validate-system     # Check system health
  python test_runner.py --unit                # Run unit tests
  python test_runner.py --integration         # Run integration tests
        """,
    )

    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    parser.add_argument(
        "--validate-system", action="store_true", help="Validate system health"
    )
    parser.add_argument("--fast", action="store_true", help="Run critical tests only")
    parser.add_argument("--report", action="store_true", help="Generate test report")

    args = parser.parse_args()

    runner = TestRunner()

    if args.all:
        runner.run_all_tests()
    elif args.unit:
        runner.test_results["Unit Tests"] = runner.run_unit_tests()
    elif args.integration:
        runner.test_results["Integration Tests"] = runner.run_integration_tests()
    elif args.performance:
        runner.test_results["Performance Tests"] = runner.run_performance_tests()
    elif args.validate_system:
        runner.test_results["System Health"] = runner.validate_system_health()
    elif args.fast:
        runner.run_fast_tests()
    else:
        # Default: run fast tests
        runner.run_fast_tests()

    # Always generate and display report
    report = runner.generate_test_report()
    print(report)

    # Save detailed report
    with open("test_report.json", "w") as f:
        json.dump(runner.test_results, f, indent=2, default=str)

    print("\n💾 Detailed report saved to: test_report.json")

    # Exit with appropriate code
    all_success = all(
        (
            result.get("success", False)
            if isinstance(result, dict)
            else all(r.get("success", False) for r in result)
        )
        for result in runner.test_results.values()
    )

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
