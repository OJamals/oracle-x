#!/usr/bin/env python3
"""
Oracle-X Unified Pipeline Runner

Consolidated main.py that supports multiple execution modes.
"""

import argparse
import json
import sys
from pathlib import Path

# Import the unified pipeline
from oracle_pipeline import OracleXPipeline


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Oracle-X Unified Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run standard pipeline
  python main.py --mode optimized          # Run optimized pipeline
  python main.py --mode advanced           # Run advanced pipeline
  python main.py --config config.json     # Run with custom config
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["standard", "optimized", "advanced"],
        default="standard",
        help="Pipeline execution mode (default: standard)"
    )  
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON)"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from {args.config}")
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")
    
    # Initialize and run pipeline
    pipeline = OracleXPipeline(mode=args.mode, config=config)
    result = pipeline.run()
    
    if result:
        print(f"\n🎉 Oracle-X {args.mode.title()} Pipeline completed successfully!")
        print(f"Results saved to: {result}")
    else:
        print(f"\n💥 Oracle-X {args.mode.title()} Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
