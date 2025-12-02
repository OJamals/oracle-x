#!/usr/bin/env python3
"""
Oracle-X Prompt Optimization CLI

Command-line interface for managing and monitoring the Oracle-X prompt optimization system.
Provides tools for analytics, experimentation, template management, and performance monitoring.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from oracle_engine.prompts.prompt_optimization import (
    get_optimization_engine,
    MarketCondition,
    PromptStrategy,
)
from oracle_engine.agent_optimized import get_optimized_agent
from oracle_engine.chains.prompt_chain import (
    get_optimization_analytics,
    evolve_prompt_templates,
)


def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  🤖 ORACLE-X PROMPT OPTIMIZATION: {title.upper()}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print formatted section"""
    print(f"\n🔹 {title}")
    print("-" * 40)


def show_analytics(days: int = 7):
    """Show comprehensive analytics"""
    print_header("Performance Analytics")

    try:
        analytics = get_optimization_analytics()
        agent = get_optimized_agent()
        performance_summary = agent.get_performance_summary(days)

        print_section("Template Performance")
        template_perfs = analytics.get("template_performance", [])
        if template_perfs:
            print(
                f"{'Template ID':<25} {'Success Rate':<12} {'Avg Latency':<12} {'Usage Count':<12}"
            )
            print("-" * 65)
            for perf in sorted(
                template_perfs, key=lambda x: x["avg_success_rate"], reverse=True
            ):
                print(
                    f"{perf['template_id']:<25} {perf['avg_success_rate']:.2%}     "
                    f"{perf['avg_latency']:.3f}s      {perf['total_usage']:<12}"
                )
        else:
            print("No template performance data available")

        print_section("Experiment Results")
        experiments = analytics.get("experiment_results", [])
        if experiments:
            print(
                f"{'Experiment ID':<15} {'Variant A':<20} {'Variant B':<20} {'Winner':<15}"
            )
            print("-" * 75)
            for exp in experiments:
                print(
                    f"{exp['experiment_id']:<15} {exp['variant_a']:<20} "
                    f"{exp['variant_b']:<20} {exp['winner'] or 'TBD':<15}"
                )
        else:
            print("No experiment results available")

        print_section("Pipeline Metrics")
        pipeline_metrics = performance_summary.get("pipeline_metrics", {})
        if pipeline_metrics.get("total_runs", 0) > 0:
            print(f"Total Runs (last {days} days): {pipeline_metrics['total_runs']}")
            print(f"Success Rate: {pipeline_metrics.get('success_rate', 0):.2%}")
            print(
                f"Average Duration: {pipeline_metrics.get('avg_duration_seconds', 0):.2f}s"
            )

            template_perf = pipeline_metrics.get("template_performance", {})
            if template_perf:
                print(f"\nTemplate Usage:")
                for template, stats in template_perf.items():
                    print(
                        f"  • {template}: {stats['usage_count']} runs, "
                        f"{stats['success_rate']:.2%} success"
                    )
        else:
            print(f"No pipeline runs in the last {days} days")

        print_section("Summary")
        print(f"Total Templates: {analytics.get('total_templates', 0)}")
        print(f"Active Experiments: {analytics.get('active_experiments', 0)}")
        print(
            f"Optimization Status: {'✅ Enabled' if get_optimized_agent().optimization_enabled else '❌ Disabled'}"
        )

    except Exception as e:
        print(f"❌ Error getting analytics: {e}")


def list_templates():
    """List all available prompt templates"""
    print_header("Available Templates")

    try:
        engine = get_optimization_engine()
        templates = engine.prompt_templates

        if not templates:
            print("No templates available")
            return

        print(f"{'Template ID':<25} {'Strategy':<15} {'Market Conditions':<30}")
        print("-" * 75)

        for template_id, template in templates.items():
            conditions = ", ".join([c.value for c in template.market_conditions])
            print(f"{template_id:<25} {template.strategy.value:<15} {conditions:<30}")

        print(f"\nTotal templates: {len(templates)}")

    except Exception as e:
        print(f"❌ Error listing templates: {e}")


def show_template_details(template_id: str):
    """Show detailed information about a specific template"""
    print_header(f"Template Details: {template_id}")

    try:
        engine = get_optimization_engine()
        template = engine.prompt_templates.get(template_id)

        if not template:
            print(f"❌ Template '{template_id}' not found")
            return

        print_section("Basic Information")
        print(f"Name: {template.name}")
        print(f"Strategy: {template.strategy.value}")
        print(
            f"Market Conditions: {', '.join([c.value for c in template.market_conditions])}"
        )
        print(f"Max Tokens: {template.max_tokens}")
        print(f"Temperature: {template.temperature}")
        print(f"Context Compression: {template.context_compression_ratio:.2f}")

        print_section("Priority Signals")
        if template.priority_signals:
            for signal in template.priority_signals:
                weight = template.signal_weights.get(signal, 0.0)
                print(f"  • {signal}: {weight:.2f}")
        else:
            print("No priority signals defined")

        print_section("System Prompt")
        print(
            template.system_prompt[:200] + "..."
            if len(template.system_prompt) > 200
            else template.system_prompt
        )

        print_section("User Prompt Template")
        print(
            template.user_prompt_template[:300] + "..."
            if len(template.user_prompt_template) > 300
            else template.user_prompt_template
        )

    except Exception as e:
        print(f"❌ Error showing template details: {e}")


def start_experiment(
    template_a: str, template_b: str, market_condition: str, duration: int = 24
):
    """Start an A/B test experiment"""
    print_header("Starting A/B Test Experiment")

    try:
        # Validate market condition
        try:
            condition = MarketCondition(market_condition.lower())
        except ValueError:
            print(f"❌ Invalid market condition: {market_condition}")
            print(f"Valid options: {', '.join([c.value for c in MarketCondition])}")
            return

        agent = get_optimized_agent()
        experiment_id = agent.start_optimization_experiment(
            template_a, template_b, condition, duration
        )

        if experiment_id and experiment_id != "Optimization not enabled":
            print(f"✅ Started experiment: {experiment_id}")
            print(f"   Variant A: {template_a}")
            print(f"   Variant B: {template_b}")
            print(f"   Market Condition: {condition.value}")
            print(f"   Duration: {duration} hours")
        else:
            print(f"❌ Failed to start experiment: {experiment_id}")

    except Exception as e:
        print(f"❌ Error starting experiment: {e}")


def run_learning_cycle(threshold: float = 0.7):
    """Run a learning cycle to evolve templates"""
    print_header("Running Learning Cycle")

    try:
        agent = get_optimized_agent()
        result = agent.run_learning_cycle(threshold)

        if result.get("evolution_successful"):
            evolved = result.get("evolved_templates", [])
            print(f"✅ Learning cycle completed successfully")
            print(f"   Evolved Templates: {len(evolved)}")
            print(f"   Performance Threshold: {threshold:.2%}")

            if evolved:
                print(f"   New Template IDs:")
                for template_id in evolved:
                    print(f"     • {template_id}")
        else:
            error = result.get("error", "Unknown error")
            print(f"❌ Learning cycle failed: {error}")

    except Exception as e:
        print(f"❌ Error running learning cycle: {e}")


def test_optimization(
    prompt: str = "Market analysis for AAPL", enable_experiments: bool = False
):
    """Test the optimization system with a sample prompt"""
    print_header("Testing Optimization System")

    try:
        agent = get_optimized_agent()

        print(f"Testing with prompt: '{prompt}'")
        print(f"Experiments enabled: {enable_experiments}")

        start_time = time.time()
        playbook, metadata = agent.oracle_agent_pipeline_optimized(
            prompt, None, enable_experiments=enable_experiments
        )
        duration = time.time() - start_time

        print_section("Results")
        print(f"Execution Time: {duration:.2f}s")
        print(
            f"Success: {'✅' if metadata['performance_metrics']['success'] else '❌'}"
        )
        print(f"Output Length: {len(playbook)} characters")

        if metadata.get("stages"):
            print_section("Stage Performance")
            for stage_name, stage_data in metadata["stages"].items():
                stage_duration = stage_data.get("duration", 0)
                print(f"  {stage_name}: {stage_duration:.2f}s")

        if metadata.get("error"):
            print(f"❌ Error: {metadata['error']}")
        elif playbook:
            print_section("Sample Output")
            print(playbook[:300] + "..." if len(playbook) > 300 else playbook)

    except Exception as e:
        print(f"❌ Error testing optimization: {e}")


def export_analytics(output_file: str, days: int = 30):
    """Export analytics to JSON file"""
    print_header("Exporting Analytics")

    try:
        analytics = get_optimization_analytics()
        agent = get_optimized_agent()
        performance_summary = agent.get_performance_summary(days)

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "days_covered": days,
            "optimization_analytics": analytics,
            "performance_summary": performance_summary,
        }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"✅ Analytics exported to: {output_file}")
        print(f"   Data covers last {days} days")

    except Exception as e:
        print(f"❌ Error exporting analytics: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Oracle-X Prompt Optimization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python oracle_optimize_cli.py analytics --days 7
  python oracle_optimize_cli.py templates list
  python oracle_optimize_cli.py templates show conservative_balanced
  python oracle_optimize_cli.py experiment start conservative_balanced aggressive_momentum bullish --duration 48
  python oracle_optimize_cli.py learning run --threshold 0.75
  python oracle_optimize_cli.py test --prompt "Analyze TSLA options" --experiments
  python oracle_optimize_cli.py export analytics.json --days 30
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analytics command
    analytics_parser = subparsers.add_parser(
        "analytics", help="Show performance analytics"
    )
    analytics_parser.add_argument(
        "--days", type=int, default=7, help="Number of days to analyze"
    )

    # Templates command
    templates_parser = subparsers.add_parser(
        "templates", help="Manage prompt templates"
    )
    templates_subparsers = templates_parser.add_subparsers(dest="templates_action")

    templates_subparsers.add_parser("list", help="List all templates")
    show_parser = templates_subparsers.add_parser("show", help="Show template details")
    show_parser.add_argument("template_id", help="Template ID to show")

    # Experiment command
    experiment_parser = subparsers.add_parser(
        "experiment", help="Manage A/B test experiments"
    )
    experiment_subparsers = experiment_parser.add_subparsers(dest="experiment_action")

    start_exp_parser = experiment_subparsers.add_parser("start", help="Start A/B test")
    start_exp_parser.add_argument("template_a", help="First template to test")
    start_exp_parser.add_argument("template_b", help="Second template to test")
    start_exp_parser.add_argument("market_condition", help="Market condition for test")
    start_exp_parser.add_argument(
        "--duration", type=int, default=24, help="Duration in hours"
    )

    # Learning command
    learning_parser = subparsers.add_parser("learning", help="Manage learning cycles")
    learning_subparsers = learning_parser.add_subparsers(dest="learning_action")

    run_learning_parser = learning_subparsers.add_parser(
        "run", help="Run learning cycle"
    )
    run_learning_parser.add_argument(
        "--threshold", type=float, default=0.7, help="Performance threshold"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test optimization system")
    test_parser.add_argument(
        "--prompt", default="Market analysis for AAPL", help="Test prompt"
    )
    test_parser.add_argument(
        "--experiments", action="store_true", help="Enable experiments"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export analytics data")
    export_parser.add_argument("output_file", help="Output JSON file")
    export_parser.add_argument(
        "--days", type=int, default=30, help="Number of days to export"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Route to appropriate function
    try:
        if args.command == "analytics":
            show_analytics(args.days)
        elif args.command == "templates":
            if args.templates_action == "list":
                list_templates()
            elif args.templates_action == "show":
                show_template_details(args.template_id)
        elif args.command == "experiment":
            if args.experiment_action == "start":
                start_experiment(
                    args.template_a,
                    args.template_b,
                    args.market_condition,
                    args.duration,
                )
        elif args.command == "learning":
            if args.learning_action == "run":
                run_learning_cycle(args.threshold)
        elif args.command == "test":
            test_optimization(args.prompt, args.experiments)
        elif args.command == "export":
            export_analytics(args.output_file, args.days)
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
