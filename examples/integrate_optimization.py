#!/usr/bin/env python3
"""
Oracle-X Optimization Integration

Integration script to enable the optimization system in the main Oracle-X pipeline.
Provides seamless integration between existing pipeline and new optimization capabilities.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

def setup_optimization_environment():
    """Setup environment variables for optimization"""
    optimization_config = {
        'ORACLE_OPTIMIZATION_ENABLED': 'true',
        'ORACLE_OPTIMIZATION_DB_PATH': 'oracle_optimization.db',
        'ORACLE_OPTIMIZATION_EXPERIMENTS_ENABLED': 'true',
        'ORACLE_OPTIMIZATION_LEARNING_ENABLED': 'true',
        'ORACLE_OPTIMIZATION_MIN_RUNS_FOR_LEARNING': '10',
        'ORACLE_OPTIMIZATION_PERFORMANCE_THRESHOLD': '0.7',
        'ORACLE_OPTIMIZATION_TEMPLATE_EVOLUTION_RATE': '0.1',
        'ORACLE_OPTIMIZATION_LOG_LEVEL': 'INFO'
    }
    
    print("🔧 Setting up optimization environment...")
    for key, value in optimization_config.items():
        os.environ[key] = value
        print(f"   {key} = {value}")
    
    print("✅ Optimization environment configured")

def test_pipeline_integration():
    """Test integration between main pipeline and optimization system"""
    print("\n🧪 Testing pipeline integration...")
    
    try:
        # Test that optimization modules can be imported
        from oracle_engine.prompt_optimization import get_optimization_engine
        from oracle_engine.agent_optimized import get_optimized_agent
        from oracle_engine.prompt_chain_optimized import get_optimization_analytics
        
        print("✅ Optimization modules imported successfully")
        
        # Test engine initialization
        engine = get_optimization_engine()
        print(f"✅ Optimization engine initialized with {len(engine.prompt_templates)} templates")
        
        # Test agent initialization
        agent = get_optimized_agent()
        print(f"✅ Optimized agent initialized (optimization enabled: {agent.optimization_enabled})")
        
        # Test analytics
        analytics = get_optimization_analytics()
        print(f"✅ Analytics system functional (total templates: {analytics.get('total_templates', 0)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def create_optimized_main_runner():
    """Create an optimized version of main.py that uses the optimization system"""
    optimized_main_content = '''#!/usr/bin/env python3
"""
Oracle-X Optimized Pipeline Runner

Enhanced version of main.py that integrates the prompt optimization system.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Setup optimization environment
os.environ['ORACLE_OPTIMIZATION_ENABLED'] = 'true'

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from oracle_engine.agent_optimized import get_optimized_agent

def main():
    """Main pipeline entry point with optimization"""
    print("🚀 Starting Oracle-X Optimized Pipeline...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Get optimized agent
        agent = get_optimized_agent()
        
        if not agent.optimization_enabled:
            print("⚠️  Optimization not enabled, falling back to standard pipeline")
            from oracle_pipeline import OracleXPipeline
            pipeline = OracleXPipeline(mode="standard")
            return pipeline.run()
        
        print(f"✅ Optimization enabled with {len(agent.prompt_optimizer.prompt_templates)} templates")
        
        # Run optimized pipeline
        prompt = "Generate comprehensive trading scenarios based on current market conditions"
        orchestrator = None
        
        try:
            from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
            orchestrator = DataFeedOrchestrator()
            print("✅ Data feed orchestrator loaded")
        except Exception as e:
            print(f"⚠️  Data feed orchestrator not available: {e}")
        
        # Execute optimized pipeline
        start_time = time.time()
        playbook, metadata = agent.oracle_agent_pipeline_optimized(
            prompt, 
            orchestrator,
            enable_experiments=True
        )
        execution_time = time.time() - start_time
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"playbooks/optimized_playbook_{timestamp}.json"
        
        # Ensure playbooks directory exists
        Path("playbooks").mkdir(exist_ok=True)
        
        # Prepare final output
        final_output = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": execution_time,
            "optimization_metadata": metadata,
            "playbook": playbook
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(final_output, f, indent=2, default=str)
        
        print(f"\\n📊 Pipeline completed successfully!")
        print(f"   Execution time: {execution_time:.2f}s")
        print(f"   Success: {'✅' if metadata['performance_metrics']['success'] else '❌'}")
        print(f"   Output saved to: {filename}")
        
        # Show optimization insights
        if metadata.get('optimization_insights'):
            insights = metadata['optimization_insights']
            print(f"\\n🧠 Optimization Insights:")
            print(f"   Template used: {insights.get('template_used', 'Unknown')}")
            print(f"   Strategy: {insights.get('strategy', 'Unknown')}")
            
            if insights.get('experiments_active'):
                print(f"   Active experiments: {len(insights['experiments_active'])}")
        
        # Show stage performance
        if metadata.get('stages'):
            print(f"\\n⏱️  Stage Performance:")
            for stage_name, stage_data in metadata['stages'].items():
                duration = stage_data.get('duration', 0)
                print(f"   {stage_name}: {duration:.2f}s")
        
        return filename
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\\n🎉 Oracle-X Optimized Pipeline completed successfully!")
        print(f"Results saved to: {result}")
    else:
        print("\\n💥 Oracle-X Optimized Pipeline failed!")
        sys.exit(1)
'''
    
    with open("main_optimized.py", 'w') as f:
        f.write(optimized_main_content)
    
    print("✅ Created main_optimized.py - optimized pipeline runner")

def setup_integration_config():
    """Create configuration files for integration"""
    
    # Environment configuration
    env_config = """# Oracle-X Optimization Configuration
# Copy to .env or source in your shell

# Core optimization settings
export ORACLE_OPTIMIZATION_ENABLED=true
export ORACLE_OPTIMIZATION_DB_PATH=oracle_optimization.db
export ORACLE_OPTIMIZATION_EXPERIMENTS_ENABLED=true
export ORACLE_OPTIMIZATION_LEARNING_ENABLED=true

# Performance thresholds
export ORACLE_OPTIMIZATION_MIN_RUNS_FOR_LEARNING=10
export ORACLE_OPTIMIZATION_PERFORMANCE_THRESHOLD=0.7
export ORACLE_OPTIMIZATION_TEMPLATE_EVOLUTION_RATE=0.1

# Logging
export ORACLE_OPTIMIZATION_LOG_LEVEL=INFO

# Advanced settings
export ORACLE_OPTIMIZATION_MAX_CONCURRENT_EXPERIMENTS=3
export ORACLE_OPTIMIZATION_EXPERIMENT_MIN_SAMPLE_SIZE=5
export ORACLE_OPTIMIZATION_GENETIC_POPULATION_SIZE=10
export ORACLE_OPTIMIZATION_GENETIC_GENERATIONS=5
"""
    
    with open("config/optimization.env", 'w') as f:
        f.write(env_config)
    
    print("✅ Created config/optimization.env configuration file")
    
    # JSON configuration
    json_config = {
        "optimization": {
            "enabled": True,
            "database_path": "oracle_optimization.db",
            "experiments_enabled": True,
            "learning_enabled": True,
            "performance_threshold": 0.7,
            "min_runs_for_learning": 10,
            "template_evolution_rate": 0.1,
            "log_level": "INFO"
        },
        "strategies": {
            "conservative": {
                "temperature": 0.3,
                "max_tokens": 2000,
                "context_compression_ratio": 0.7
            },
            "balanced": {
                "temperature": 0.5,
                "max_tokens": 3000,
                "context_compression_ratio": 0.8
            },
            "aggressive": {
                "temperature": 0.7,
                "max_tokens": 4000,
                "context_compression_ratio": 0.9
            }
        },
        "market_conditions": {
            "bullish": ["positive_sentiment", "upward_trends", "growth_signals"],
            "bearish": ["negative_sentiment", "downward_trends", "risk_signals"],
            "volatile": ["high_volatility", "mixed_signals", "uncertainty"],
            "stable": ["low_volatility", "consistent_trends", "steady_signals"]
        }
    }
    
    with open("config/optimization_config.json", 'w') as f:
        json.dump(json_config, f, indent=2)
    
    print("✅ Created config/optimization_config.json configuration file")

def create_quick_start_script():
    """Create a quick start script for the optimization system"""
    quick_start_content = '''#!/bin/bash
"""
Oracle-X Optimization Quick Start

Run this script to quickly set up and test the optimization system.
"""

echo "🚀 Oracle-X Optimization Quick Start"
echo "===================================="

# Set up environment
echo "🔧 Setting up environment..."
source config/optimization.env 2>/dev/null || echo "⚠️  config/optimization.env not found, using defaults"

# Test system
echo "🧪 Testing optimization system..."
python oracle_optimize_cli.py test --prompt "Quick test of optimization system"

if [ $? -eq 0 ]; then
    echo "✅ Optimization system test passed!"
else
    echo "❌ Optimization system test failed!"
    exit 1
fi

# Show analytics
echo "📊 Current system analytics..."
python oracle_optimize_cli.py analytics --days 1

# List templates
echo "📝 Available templates..."
python oracle_optimize_cli.py templates list

echo ""
echo "🎉 Quick start completed!"
echo "Next steps:"
echo "  1. Run optimized pipeline: python main_optimized.py"
echo "  2. Monitor performance: python oracle_optimize_cli.py analytics"
echo "  3. Start experiments: python oracle_optimize_cli.py experiment start [template_a] [template_b] [condition]"
echo ""
'''
    
    with open("quick_start_optimization.sh", 'w') as f:
        f.write(quick_start_content)
    
    # Make executable
    os.chmod("quick_start_optimization.sh", 0o755)
    
    print("✅ Created quick_start_optimization.sh script")

def generate_integration_report():
    """Generate a comprehensive integration report"""
    report = {
        "integration_timestamp": datetime.now().isoformat(),
        "system_status": {},
        "files_created": [],
        "configuration": {},
        "next_steps": []
    }
    
    # Check system status
    try:
        setup_optimization_environment()
        test_result = test_pipeline_integration()
        report["system_status"]["integration_test"] = "passed" if test_result else "failed"
        report["system_status"]["optimization_ready"] = test_result
    except Exception as e:
        report["system_status"]["integration_test"] = "failed"
        report["system_status"]["error"] = str(e)
    
    # List created files
    created_files = [
        "oracle_optimize_cli.py",
        "main_optimized.py", 
        "config/optimization.env",
        "config/optimization_config.json",
        "quick_start_optimization.sh"
    ]
    
    for file in created_files:
        if os.path.exists(file):
            report["files_created"].append({
                "filename": file,
                "size_bytes": os.path.getsize(file),
                "created": True
            })
    
    # Configuration summary
    report["configuration"] = {
        "optimization_enabled": os.getenv('ORACLE_OPTIMIZATION_ENABLED', 'false') == 'true',
        "experiments_enabled": os.getenv('ORACLE_OPTIMIZATION_EXPERIMENTS_ENABLED', 'false') == 'true',
        "learning_enabled": os.getenv('ORACLE_OPTIMIZATION_LEARNING_ENABLED', 'false') == 'true',
        "database_path": os.getenv('ORACLE_OPTIMIZATION_DB_PATH', 'oracle_optimization.db')
    }
    
    # Next steps
    report["next_steps"] = [
        "Source config/optimization.env file to enable environment variables",
        "Run './quick_start_optimization.sh' to test the system",
        "Execute 'python main_optimized.py' to run optimized pipeline",
        "Monitor performance with 'python oracle_optimize_cli.py analytics'",
        "Start A/B experiments with 'python oracle_optimize_cli.py experiment start'"
    ]
    
    # Save report
    with open("integration_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("✅ Integration report saved to integration_report.json")
    return report

def main():
    """Main integration script"""
    parser = argparse.ArgumentParser(description="Oracle-X Optimization Integration")
    parser.add_argument('--setup', action='store_true', help='Setup integration files')
    parser.add_argument('--test', action='store_true', help='Test integration')
    parser.add_argument('--report', action='store_true', help='Generate integration report')
    parser.add_argument('--all', action='store_true', help='Run all integration steps')
    
    args = parser.parse_args()
    
    if not any([args.setup, args.test, args.report, args.all]):
        parser.print_help()
        return
    
    print("🔄 Oracle-X Optimization Integration")
    print("=====================================")
    
    try:
        if args.setup or args.all:
            print("\n📁 Creating integration files...")
            setup_optimization_environment()
            create_optimized_main_runner()
            setup_integration_config()
            create_quick_start_script()
            print("✅ Integration files created successfully")
        
        if args.test or args.all:
            print("\n🧪 Testing integration...")
            test_result = test_pipeline_integration()
            if test_result:
                print("✅ Integration test passed")
            else:
                print("❌ Integration test failed")
        
        if args.report or args.all:
            print("\n📊 Generating integration report...")
            report = generate_integration_report()
            print("✅ Integration report generated")
            
            # Summary
            print("\n📋 Integration Summary:")
            print(f"   Status: {'✅ Ready' if report['system_status'].get('optimization_ready') else '❌ Not Ready'}")
            print(f"   Files Created: {len(report['files_created'])}")
            print(f"   Configuration: {'✅ Enabled' if report['configuration']['optimization_enabled'] else '❌ Disabled'}")
            
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
