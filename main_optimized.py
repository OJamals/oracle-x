#!/usr/bin/env python3
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
            from main import main as standard_main
            return standard_main()
        
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
        
        print(f"\n📊 Pipeline completed successfully!")
        print(f"   Execution time: {execution_time:.2f}s")
        print(f"   Success: {'✅' if metadata['performance_metrics']['success'] else '❌'}")
        print(f"   Output saved to: {filename}")
        
        # Show optimization insights
        if metadata.get('optimization_insights'):
            insights = metadata['optimization_insights']
            print(f"\n🧠 Optimization Insights:")
            print(f"   Template used: {insights.get('template_used', 'Unknown')}")
            print(f"   Strategy: {insights.get('strategy', 'Unknown')}")
            
            if insights.get('experiments_active'):
                print(f"   Active experiments: {len(insights['experiments_active'])}")
        
        # Show stage performance
        if metadata.get('stages'):
            print(f"\n⏱️  Stage Performance:")
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
        print(f"\n🎉 Oracle-X Optimized Pipeline completed successfully!")
        print(f"Results saved to: {result}")
    else:
        print("\n💥 Oracle-X Optimized Pipeline failed!")
        sys.exit(1)
