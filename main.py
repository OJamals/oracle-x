#!/usr/bin/env python3
"""
Oracle-X Unified Pipeline Runner

Consolidated main.py that supports multiple execution modes:
- standard: Original Oracle-X pipeline (default)
- enhanced: Enhanced pipeline with options analysis
- optimized: Pipeline with prompt optimization system

Usage:
    python main.py [--mode standard|enhanced|optimized] [--config config.json]
"""

import argparse
import warnings
warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated*",
    category=FutureWarning
)

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Memory-efficient processing import with fallback
try:
    from core.memory_processor import optimize_dataframe_memory
    MEMORY_PROCESSOR_AVAILABLE = True
except ImportError:
    MEMORY_PROCESSOR_AVAILABLE = False
    optimize_dataframe_memory = None

# Async I/O utilities import with fallback
try:
    from core.async_io_utils import get_async_io_manager
    ASYNC_IO_AVAILABLE = True
except ImportError:
    ASYNC_IO_AVAILABLE = False
    get_async_io_manager = None

# Core Oracle engine imports
from oracle_engine.agent import oracle_agent_pipeline

# Data feeds
try:
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
    from data_feeds.financial_calculator import FinancialCalculator
    orchestrator_available = True
except Exception:
    DataFeedOrchestrator = None
    FinancialCalculator = None
    orchestrator_available = False

# Enhanced pipeline imports (optional)
try:
    # All enhanced pipeline imports commented out - unused
    pass
    enhanced_options_available = True
except Exception:
    enhanced_options_available = False

# Optimized pipeline imports (optional)
try:
    from oracle_engine.agent_optimized import get_optimized_agent
    optimization_available = True
except Exception:
    optimization_available = False

# Configure logging
import logging as _logging
import sys as _sys
try:
    _root_logger = _logging.getLogger()
    if not any(isinstance(h, _logging.StreamHandler) for h in list(_root_logger.handlers)):
        target_stream = _sys.__stdout__ if hasattr(_sys, "__stdout__") and _sys.__stdout__ is not None else _sys.stdout
        _sh = _logging.StreamHandler(target_stream)
        _sh.setLevel(_logging.INFO)
        _sh.setFormatter(_logging.Formatter("%(message)s"))
        _root_logger.addHandler(_sh)
    _root_logger.setLevel(_logging.INFO)
except Exception:
    pass

class OracleXPipeline:
    """Unified Oracle-X pipeline supporting multiple execution modes"""
    
    def __init__(self, mode: str = "standard", config: Optional[Dict] = None):
        self.mode = mode
        self.config = config or {}
        self.orchestrator = None
        self.async_io_manager = None
        self._init_orchestrator()
        self._init_async_io()
    
    def _init_orchestrator(self):
        """Initialize data feed orchestrator if available"""
        if orchestrator_available and DataFeedOrchestrator is not None:
            try:
                self.orchestrator = DataFeedOrchestrator()
                print("✅ Data feed orchestrator loaded")
            except Exception as e:
                print(f"⚠️  Data feed orchestrator not available: {e}")
                self.orchestrator = None
        else:
            print("⚠️  Data feed orchestrator not available")
            self.orchestrator = None
    
    def _init_async_io(self):
        """Initialize async I/O manager if available"""
        if ASYNC_IO_AVAILABLE and get_async_io_manager is not None:
            try:
                # For sync context, we'll initialize on first async use
                self.async_io_available = True
                print("✅ Async I/O manager available")
            except Exception as e:
                print(f"⚠️  Async I/O manager not available: {e}")
                self.async_io_available = False
        else:
            print("⚠️  Async I/O manager not available")
            self.async_io_available = False

    async def _get_async_io_manager(self):
        """Get async I/O manager instance for async operations"""
        if not self.async_io_available or get_async_io_manager is None:
            return None
        try:
            return await get_async_io_manager()
        except Exception as e:
            print(f"⚠️  Failed to get async I/O manager: {e}")
            return None

    async def _save_pipeline_results_async(self, filename: str, data: Dict[str, Any]) -> bool:
        """Save pipeline results asynchronously if available, fallback to sync"""
        io_manager = await self._get_async_io_manager()
        if io_manager and io_manager.file:
            try:
                success = await io_manager.file.write_json(filename, data)
                if success:
                    print(f"📁 Results saved asynchronously to: {filename}")
                    return True
            except Exception as e:
                print(f"⚠️  Async save failed, falling back to sync: {e}")
        
        # Fallback to sync
        try:
            # Ensure playbooks directory exists
            Path("playbooks").mkdir(exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"📁 Results saved synchronously to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return False

    def fetch_price_history(self, ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Fetch historical price data prioritizing orchestrator feeds, fallback to yfinance direct."""
        if self.orchestrator is not None:
            try:
                md = self.orchestrator.get_market_data(ticker, period=f"{days}d", interval="1d")
                if md and isinstance(md.data, pd.DataFrame) and not md.data.empty:
                    # Optimize memory usage of the DataFrame
                    if MEMORY_PROCESSOR_AVAILABLE and optimize_dataframe_memory:
                        return optimize_dataframe_memory(md.data)
                    return md.data
            except Exception as e:
                print(f"[WARN] Orchestrator market data failed for {ticker}: {e}")
        
        # Fallback: direct yfinance
        end = datetime.now()
        start = end - timedelta(days=days)
        try:
            df = yf.download(
                ticker,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                progress=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Optimize memory usage of the DataFrame
                if MEMORY_PROCESSOR_AVAILABLE and optimize_dataframe_memory:
                    return optimize_dataframe_memory(df)
                return df
        except Exception as e:
            print(f"[WARN] Failed to fetch price history for {ticker}: {e}")
        return None

    def plot_price_chart(self, ticker: str, image_path: str, days: int = 60):
        """Generate price chart for ticker"""
        df = self.fetch_price_history(ticker, days)
        if df is None or df.empty:
            print(f"[WARN] No price data for {ticker}, skipping chart.")
            return None
        
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Close'], label=f"{ticker} Close", color='royalblue')
        plt.title(f"{ticker} Price Chart (Last {days} Days)", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price ($)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()
        return image_path

    def run_standard_pipeline(self):
        """Run the standard Oracle-X pipeline"""
        print("🚀 Starting Oracle-X Standard Pipeline...")
        
        try:
            # Standard Oracle pipeline execution
            prompt = "Generate comprehensive trading scenarios based on current market conditions"
            
            # Generate a chart if possible (placeholder for now)
            chart_image_b64 = None
            
            # Run Oracle agent pipeline
            start_time = time.time()
            
            # Use the core oracle agent pipeline
            result = oracle_agent_pipeline(prompt, chart_image_b64)
            
            execution_time = time.time() - start_time
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"playbooks/standard_playbook_{timestamp}.json"
            
            # Prepare final output
            final_output = {
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": execution_time,
                "pipeline_mode": "standard",
                "playbook": result
            }
            
            # Use async file saving if available
            import asyncio
            success = asyncio.run(self._save_pipeline_results_async(filename, final_output))
            
            if success:
                print("\n📊 Standard Pipeline completed successfully!")
                print(f"   Execution time: {execution_time:.2f}s")
                print(f"   Output saved to: {filename}")
                return filename
            else:
                print("❌ Failed to save pipeline results")
                return None
            
        except Exception as e:
            print(f"❌ Standard pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_enhanced_pipeline(self):
        """Run the enhanced Oracle-X pipeline with options analysis"""
        print("🚀 Starting Oracle-X Enhanced Pipeline...")
        
        if not enhanced_options_available:
            print("❌ Enhanced options pipeline not available")
            print("ℹ️  Falling back to standard pipeline")
            return self.run_standard_pipeline()
        
        try:
            # Enhanced pipeline with options analysis
            prompt = "Generate enhanced trading scenarios with options analysis"
            chart_image_b64 = None
            
            # Run enhanced oracle agent pipeline
            start_time = time.time()
            result = oracle_agent_pipeline(prompt, chart_image_b64)
            execution_time = time.time() - start_time
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"playbooks/enhanced_playbook_{timestamp}.json"
            
            # Prepare final output
            final_output = {
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": execution_time,
                "pipeline_mode": "enhanced", 
                "playbook": result
            }
            
            # Use async file saving if available
            import asyncio
            success = asyncio.run(self._save_pipeline_results_async(filename, final_output))
            
            if success:
                print("\n📊 Enhanced Pipeline completed successfully!")
                print(f"   Execution time: {execution_time:.2f}s")
                print(f"   Output saved to: {filename}")
                return filename
            else:
                print("❌ Failed to save pipeline results")
                return None
            
        except Exception as e:
            print(f"❌ Enhanced pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_optimized_pipeline(self):
        """Run the optimized Oracle-X pipeline with prompt optimization"""
        print("🚀 Starting Oracle-X Optimized Pipeline...")
        
        if not optimization_available:
            print("❌ Optimization system not available")
            print("ℹ️  Falling back to standard pipeline")
            return self.run_standard_pipeline()
        
        # Setup optimization environment
        os.environ['ORACLE_OPTIMIZATION_ENABLED'] = 'true'
        
        try:
            # Get optimized agent
            if not optimization_available:
                print("⚠️  Optimization not available, falling back to standard pipeline")
                return self.run_standard_pipeline()
                
            # Import here to avoid unbound variable issues
            from oracle_engine.agent_optimized import get_optimized_agent
            agent = get_optimized_agent()
            
            if not agent.optimization_enabled:
                print("⚠️  Optimization not enabled, falling back to standard pipeline")
                return self.run_standard_pipeline()
            
            print("✅ Optimization enabled")
            
            # Run optimized pipeline
            prompt = "Generate comprehensive trading scenarios based on current market conditions"
            
            # Execute optimized pipeline
            start_time = time.time()
            playbook, metadata = agent.oracle_agent_pipeline_optimized(
                prompt, 
                None,  # chart_image_b64
                enable_experiments=True
            )
            execution_time = time.time() - start_time
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"playbooks/optimized_playbook_{timestamp}.json"
            
            # Prepare final output
            final_output = {
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": execution_time,
                "optimization_metadata": metadata,
                "playbook": playbook
            }
            
            # Use async file saving if available
            import asyncio
            success = asyncio.run(self._save_pipeline_results_async(filename, final_output))
            
            if success:
                print("\n📊 Pipeline completed successfully!")
                print(f"   Execution time: {execution_time:.2f}s")
                print(f"   Success: {'✅' if metadata['performance_metrics']['success'] else '❌'}")
                print(f"   Output saved to: {filename}")
                return filename
            else:
                print("❌ Failed to save pipeline results")
                return None
            
        except Exception as e:
            print(f"❌ Optimized pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run(self):
        """Run the pipeline in the specified mode"""
        print(f"🎯 Oracle-X Pipeline Mode: {self.mode.upper()}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        if self.mode == "standard":
            return self.run_standard_pipeline()
        elif self.mode == "enhanced":
            return self.run_enhanced_pipeline()
        elif self.mode == "optimized":
            return self.run_optimized_pipeline()
        else:
            print(f"❌ Unknown mode: {self.mode}")
            return None

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Oracle-X Unified Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run standard pipeline
  python main.py --mode enhanced           # Run enhanced pipeline
  python main.py --mode optimized          # Run optimized pipeline
  python main.py --config config.json     # Run with custom config
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["standard", "enhanced", "optimized"],
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
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from {args.config}")
        except Exception as e:
            print(f"⚠️  Failed to load config file: {e}")
    
    # Initialize and run pipeline
    pipeline = OracleXPipeline(mode=args.mode, config=config)
    result = pipeline.run()
    
    if result:
        print(f"\n🎉 Oracle-X {args.mode.title()} Pipeline completed successfully!")
        if isinstance(result, str):
            print(f"Results saved to: {result}")
    else:
        print(f"\n💥 Oracle-X {args.mode.title()} Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Dashboard integration function
def run_oracle_pipeline(prompt_text: str) -> Dict:
    """Function for dashboard integration - runs standard pipeline with given prompt"""
    pipeline = OracleXPipeline(mode="standard")
    try:
        # Run the pipeline
        result_file = pipeline.run_standard_pipeline()
        if result_file and os.path.exists(result_file):
            # Load the results
            with open(result_file, 'r') as f:
                results = json.load(f)

            # Add current date and logs for dashboard compatibility
            results["date"] = datetime.now().strftime("%Y-%m-%d")
            results["logs"] = f"Pipeline executed successfully at {datetime.now().isoformat()}"

            return results
        else:
            raise Exception("Pipeline execution failed - no results file generated")
    except Exception as e:
        print(f"Dashboard pipeline execution failed: {e}")
        # Return minimal structure for dashboard compatibility
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "playbook": {"trades": [], "tomorrows_tape": "Pipeline execution failed"},
            "logs": f"Error: {str(e)}"
        }
