#!/usr/bin/env python3
"""
🚀 ORACLE-X Unified CLI Interface

Single entry point for options analysis, optimization analytics, validation,
pipeline orchestration, and test execution.

Available Commands:
    options     - Options analysis and trading commands
    optimize    - Prompt optimization and performance analytics
    validate    - System validation and testing
    pipeline    - Pipeline management and execution
    test        - Test execution and reporting

Usage Examples:
    oracle_cli.py options analyze AAPL --verbose
    oracle_cli.py optimize analytics --days 7
    oracle_cli.py validate system --comprehensive
    oracle_cli.py pipeline run --mode enhanced
    oracle_cli.py test --all
"""

import argparse
import json
import importlib
import sys
import subprocess
import warnings

# Import common utilities
from utils.common import (
    setup_project_path,
    CLIFormatter,
    setup_logging,
    get_project_root,
)

# Setup project path and logging
setup_project_path()
project_root = get_project_root()
logger = setup_logging("oracle-cli")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# Use CLIFormatter for consistent output
def print_header(title: str):
    """Print formatted header"""
    print(CLIFormatter.header(title))


def print_section(title: str):
    """Print formatted section"""
    print(CLIFormatter.section(title))


def print_error(message: str):
    """Print formatted error message"""
    print(CLIFormatter.error(message))


def print_success(message: str):
    """Print formatted success message"""
    print(CLIFormatter.success(message))


def print_info(message: str):
    """Print formatted info message"""
    print(CLIFormatter.info(message))


def _check_component(name: str, loader):
    """Attempt to load a component and return status with optional error."""
    try:
        loader()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _run_command(
    cmd, description: str, *, background: bool = False, verbose: bool = False
) -> bool:
    """Run a subprocess command with consistent logging."""
    try:
        if background:
            subprocess.Popen(cmd, cwd=project_root)
            print_success(f"{description} started in background")
            return True

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=verbose,
            text=True,
        )
        if result.returncode == 0:
            print_success(f"{description} completed successfully")
        else:
            print_error(f"{description} failed with exit code {result.returncode}")

        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)

        return result.returncode == 0
    except Exception as exc:  # noqa: BLE001
        print_error(f"{description} failed: {exc}")
        return False


# ========================== OPTIONS COMMANDS ==========================


def handle_options_analyze(args):
    """Handle options analysis command"""
    print_header("Options Analysis")

    try:
        from oracle_options_pipeline import create_pipeline

        # Create pipeline with dictionary config
        config = {"risk_tolerance": "moderate", "min_opportunity_score": 60.0}

        pipeline = create_pipeline(config)

        print_info(f"Analyzing options for {args.symbol}")

        # Get recommendations
        recommendations = pipeline.analyze_ticker(args.symbol)

        if not recommendations:
            print_info(f"No suitable options found for {args.symbol}")
            return

        # Display results using the correct attributes
        for i, rec in enumerate(recommendations[: args.limit], 1):
            print(f"\n📊 Option #{i}: {rec.symbol}")
            print(f"   Type: {rec.contract.option_type.value.upper()}")
            print(f"   Strike: ${rec.contract.strike:.2f}")
            print(f"   Expiry: {rec.contract.expiry.strftime('%Y-%m-%d')}")
            print(f"   Opportunity Score: {rec.opportunity_score:.1f}/100")

            if args.verbose:
                print(f"   ML Confidence: {rec.ml_confidence:.1%}")
                print(f"   Entry Price: ${rec.entry_price:.2f}")
                print(f"   Target Price: ${rec.target_price:.2f}")
                print(f"   Risk/Reward: {rec.risk_reward_ratio:.2f}")

        if args.output:
            # Convert to dict for JSON serialization using to_dict method
            data = [rec.to_dict() for rec in recommendations]
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print_success(f"Results saved to {args.output}")

    except Exception as e:
        print_error(f"Options analysis failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


def handle_options_scan(args):
    """Handle options market scan command"""
    print_header("Options Market Scan")
    print_info("Market scanning functionality coming soon...")


# ========================== OPTIMIZATION COMMANDS ==========================


def handle_optimize_analytics(args):
    """Handle optimization analytics command"""
    print_header("Optimization Analytics")

    try:
        from oracle_engine.agent_optimized import get_optimized_agent
        from oracle_engine.chains.prompt_chain import get_optimization_analytics

        analytics = get_optimization_analytics()
        agent = get_optimized_agent()
        performance_summary = agent.get_performance_summary(args.days)

        print_section("Template Performance")
        template_perfs = analytics.get("template_performance", [])
        if template_perfs:
            print(
                f"{'Template ID':<30} {'Success Rate':<12} {'Avg Latency':<12} {'Usage':<8}"
            )
            print("-" * 70)
            for perf in sorted(
                template_perfs, key=lambda x: x.get("avg_success_rate", 0), reverse=True
            ):
                success_rate = perf.get("avg_success_rate", 0)
                latency = perf.get("avg_latency_ms", 0)
                usage = perf.get("usage_count", 0)
                print(
                    f"{perf['template_id']:<30} {success_rate:.2%}        "
                    f"{latency:.1f}ms       {usage}"
                )
        else:
            print_info("No template performance data available")

        print_section("System Performance")
        if performance_summary:
            total_requests = performance_summary.get("total_requests", 0)
            success_rate = performance_summary.get("success_rate", 0)
            avg_latency = performance_summary.get("avg_latency_ms", 0)

            print(f"Total Requests: {total_requests}")
            print(f"Success Rate: {success_rate:.2%}")
            print(f"Average Latency: {avg_latency:.1f}ms")
        else:
            print_info("No performance summary available")

    except Exception as e:
        print_error(f"Analytics failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


def handle_optimize_templates(args):
    """Handle optimization templates command"""
    print_header("Template Management")
    print_info("Template management functionality coming soon...")


# ========================== VALIDATION COMMANDS ==========================


def handle_validate_system(args):
    """Handle system validation command"""
    print_header("System Validation")

    try:
        # Import major system components
        print_info("Validating system components...")

        checks = [
            (
                "DataFeedOrchestrator",
                lambda: importlib.import_module(
                    "data_feeds.data_feed_orchestrator"
                ).DataFeedOrchestrator(),
            ),
            ("Configuration", lambda: importlib.import_module("core.config").config),
            (
                "PromptChain",
                lambda: importlib.import_module("oracle_engine.prompt_chain"),
            ),
            (
                "MLModelManager",
                lambda: importlib.import_module(
                    "oracle_engine.ml_model_manager"
                ).MLModelManager,
            ),
        ]

        components = [
            (name, *_check_component(name, loader)) for name, loader in checks
        ]

        # Display results
        print_section("Component Status")
        success_count = 0
        for name, status, error in components:
            if status:
                print(f"✅ {name}")
                success_count += 1
            else:
                print(f"❌ {name}: {error}")

        success_rate = (success_count / len(components)) * 100
        print(
            f"\n🎯 System Health: {success_rate:.1f}% ({success_count}/{len(components)})"
        )

        if args.comprehensive:
            print_info("Running comprehensive validation...")
            _run_command(
                [sys.executable, "test_runner.py", "--validate-system"],
                "Comprehensive validation",
                verbose=True,
            )

    except Exception as e:
        print_error(f"System validation failed: {e}")


# ========================== PIPELINE COMMANDS ==========================


def handle_pipeline_run(args):
    """Handle pipeline execution command"""
    print_header("Pipeline Execution")

    mode_files = {
        "standard": "main.py",
        "signals": "signals_runner.py",
        "options": "oracle_options_pipeline.py",
    }

    if args.mode == "all":
        # Run all pipelines sequentially
        print_info("Running all pipelines sequentially...")
        for mode, script in mode_files.items():
            print_section(f"Running {mode} pipeline")
            _run_command(
                [sys.executable, script],
                f"{mode} pipeline",
                verbose=args.verbose,
            )
        return

    script = mode_files.get(args.mode)
    if not script:
        print_error(f"Unknown pipeline mode: {args.mode}")
        print_info(f"Available modes: {', '.join(mode_files.keys())}, all")
        return

    print_info(f"Running {args.mode} pipeline ({script})")

    cmd = [sys.executable, script]
    _run_command(
        cmd,
        f"{args.mode} pipeline",
        background=args.background,
        verbose=args.verbose,
    )


def handle_pipeline_status(args):
    """Handle pipeline status command"""
    print_header("Pipeline Status")
    print_info("Pipeline status monitoring coming soon...")


# ========================== TEST COMMANDS ==========================


def handle_test(args):
    """Handle test execution command"""
    print_header("Test Execution")

    try:
        cmd = [sys.executable, "test_runner.py"]

        if args.all:
            cmd.append("--all")
        elif args.unit:
            cmd.append("--unit")
        elif args.integration:
            cmd.append("--integration")
        elif args.performance:
            cmd.append("--performance")
        elif args.fast:
            cmd.append("--fast")
        else:
            cmd.append("--validate-system")  # Default

        if args.report:
            cmd.append("--report")

        print_info(f"Running tests: {' '.join(cmd[2:])}")

        _run_command(cmd, "Tests", verbose=args.verbose)

    except Exception as e:
        print_error(f"Test execution failed: {e}")


# ========================== MAIN CLI SETUP ==========================


def create_parser():
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        description="🚀 ORACLE-X Unified CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s options analyze AAPL --verbose
  %(prog)s optimize analytics --days 7
  %(prog)s validate system --comprehensive
  %(prog)s pipeline run --mode enhanced
  %(prog)s test --all --report

For more help on specific commands:
  %(prog)s options --help
  %(prog)s optimize --help
  %(prog)s validate --help
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Options commands
    options_parser = subparsers.add_parser(
        "options", help="Options analysis and trading"
    )
    options_subs = options_parser.add_subparsers(dest="options_command")

    # Options analyze
    analyze_parser = options_subs.add_parser(
        "analyze", help="Analyze options for a symbol"
    )
    analyze_parser.add_argument("symbol", help="Stock symbol to analyze")
    analyze_parser.add_argument(
        "--limit", type=int, default=5, help="Max recommendations (default: 5)"
    )
    analyze_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    analyze_parser.add_argument("--output", help="Save results to JSON file")

    # Options scan
    scan_parser = options_subs.add_parser("scan", help="Scan market for opportunities")
    scan_parser.add_argument("--sector", help="Focus on specific sector")
    scan_parser.add_argument("--strategy", help="Filter by strategy type")

    # Optimization commands
    opt_parser = subparsers.add_parser(
        "optimize", help="Prompt optimization and analytics"
    )
    opt_subs = opt_parser.add_subparsers(dest="optimize_command")

    # Optimization analytics
    analytics_parser = opt_subs.add_parser(
        "analytics", help="Show optimization analytics"
    )
    analytics_parser.add_argument(
        "--days", type=int, default=7, help="Days of data (default: 7)"
    )
    analytics_parser.add_argument("--export", help="Export to JSON file")
    analytics_parser.add_argument(
        "--verbose", action="store_true", help="Verbose output"
    )

    # Optimization templates
    templates_parser = opt_subs.add_parser(
        "templates", help="Manage optimization templates"
    )
    templates_parser.add_argument(
        "action", choices=["list", "show", "create"], help="Template action"
    )
    templates_parser.add_argument("name", nargs="?", help="Template name")

    # Validation commands
    val_parser = subparsers.add_parser("validate", help="System validation and testing")
    val_subs = val_parser.add_subparsers(dest="validate_command")

    # System validation
    sys_parser = val_subs.add_parser("system", help="Validate system health")
    sys_parser.add_argument(
        "--comprehensive", action="store_true", help="Run comprehensive validation"
    )
    sys_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Pipeline commands
    pipe_parser = subparsers.add_parser(
        "pipeline", help="Pipeline management and execution"
    )
    pipe_subs = pipe_parser.add_subparsers(dest="pipeline_command")

    # Pipeline run
    run_parser = pipe_subs.add_parser("run", help="Run a pipeline")
    run_parser.add_argument(
        "--mode",
        choices=["standard", "signals", "options", "all"],
        default="standard",
        help="Pipeline mode (default: standard)",
    )
    run_parser.add_argument(
        "--background", action="store_true", help="Run in background"
    )
    run_parser.add_argument(
        "--verbose", action="store_true", help="Verbose output for all mode"
    )

    # Pipeline status
    pipe_subs.add_parser("status", help="Check pipeline status")

    # Test commands
    test_parser = subparsers.add_parser("test", help="Test execution and reporting")
    test_group = test_parser.add_mutually_exclusive_group()
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--unit", action="store_true", help="Run unit tests only")
    test_group.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    test_group.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    test_group.add_argument(
        "--fast", action="store_true", help="Run fast/critical tests only"
    )
    test_parser.add_argument(
        "--report", action="store_true", help="Generate detailed report"
    )
    test_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        print_header("Welcome")
        parser.print_help()
        return

    try:
        # Route commands to handlers
        if args.command == "options":
            if args.options_command == "analyze":
                handle_options_analyze(args)
            elif args.options_command == "scan":
                handle_options_scan(args)
            else:
                print_error("Please specify an options subcommand (analyze, scan)")

        elif args.command == "optimize":
            if args.optimize_command == "analytics":
                handle_optimize_analytics(args)
            elif args.optimize_command == "templates":
                handle_optimize_templates(args)
            else:
                print_error(
                    "Please specify an optimization subcommand (analytics, templates)"
                )

        elif args.command == "validate":
            if args.validate_command == "system":
                handle_validate_system(args)
            else:
                print_error("Please specify a validation subcommand (system)")

        elif args.command == "pipeline":
            if args.pipeline_command == "run":
                handle_pipeline_run(args)
            elif args.pipeline_command == "status":
                handle_pipeline_status(args)
            else:
                print_error("Please specify a pipeline subcommand (run, status)")

        elif args.command == "test":
            handle_test(args)

    except KeyboardInterrupt:
        print_info("\nOperation cancelled by user")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
