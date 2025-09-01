#!/usr/bin/env python3
"""
Optimized Test Runner for Oracle-X

Provides fast test execution options with performance optimizations.
Separates unit tests from slow integration tests for better developer experience.
"""

import sys
import subprocess
import argparse

def run_command(cmd):
    """Run command and return success status"""
    print(f"🚀 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Oracle-X Test Runner")
    parser.add_argument("--fast", action="store_true", 
                       help="Run only fast unit tests (default)")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests including slow integration tests")
    parser.add_argument("--integration", action="store_true",
                       help="Run only integration tests")
    parser.add_argument("--ml", action="store_true",
                       help="Run only ML tests (very slow)")
    parser.add_argument("--unit", action="store_true",
                       help="Run only unit tests")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel execution")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Test timeout in seconds (default: 30)")
    parser.add_argument("tests", nargs="*",
                       help="Specific test files or patterns to run")
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add timeout
    cmd.extend(["--timeout", str(args.timeout)])
    
    # Add parallel processing unless disabled
    if not args.no_parallel:
        cmd.extend(["-n", "auto"])
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Determine which tests to run
    if args.all:
        print("🔄 Running ALL tests (including slow integration tests)")
        # Run all tests
    elif args.integration:
        print("🌐 Running INTEGRATION tests only")
        cmd.extend(["-m", "integration"])
    elif args.ml:
        print("🧠 Running ML tests only (this will be slow)")
        cmd.extend(["-m", "ml"])
    elif args.unit:
        print("⚡ Running UNIT tests only")
        cmd.extend(["-m", "unit"])
    else:
        # Default: exclude slow tests
        print("⚡ Running FAST tests only (excluding integration and ML tests)")
        cmd.extend(["-m", "not integration and not ml and not slow"])
    
    # Add specific test files if provided
    if args.tests:
        cmd.extend(args.tests)
    
    # Run the tests
    success = run_command(cmd)
    
    if success:
        print("✅ Tests completed successfully!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
