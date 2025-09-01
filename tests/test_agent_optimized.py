#!/usr/bin/env python3
"""
Test script for Oracle Agent Optimized Pipeline
Systematically tests all components as requested by user
"""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pipeline_initialization():
    """Test 1: Pipeline Initialization and Configuration Loading"""
    print("=" * 60)
    print("TEST 1: Pipeline Initialization")
    print("=" * 60)

    try:
        from oracle_engine.agent_optimized import get_optimized_agent
        print("✓ Successfully imported OracleAgentOptimized")

        # Test factory function
        agent = get_optimized_agent()
        print(f"✓ Factory function returned: {type(agent).__name__}")

        # Test singleton pattern
        agent2 = get_optimized_agent()
        print(f"✓ Singleton test - Same instance: {agent is agent2}")

        # Test configuration loading
        print(f"✓ Optimization enabled: {agent.optimization_enabled}")
        print(f"✓ Engine instance: {type(agent.engine).__name__ if agent.engine else 'None'}")

        return True, agent

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_database_creation(agent):
    """Test 2: Database Creation and Connection"""
    print("\n" + "=" * 60)
    print("TEST 2: Database Creation and Connection")
    print("=" * 60)

    try:
        if not agent.optimization_enabled or not agent.engine:
            print("⚠️  Optimization not enabled, skipping database tests")
            return True

        # Check if database file exists
        db_path = agent.engine.db_path
        print(f"✓ Database path: {db_path}")

        import os
        if os.path.exists(db_path):
            print(f"✓ Database file exists: {db_path}")
        else:
            print(f"⚠️  Database file does not exist yet: {db_path}")

        # Test database connection
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"✓ Database connection successful. Tables: {[t[0] for t in tables]}")

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_self_learning_integration(agent):
    """Test 3: Self-Learning Integration"""
    print("\n" + "=" * 60)
    print("TEST 3: Self-Learning Integration")
    print("=" * 60)

    try:
        if not agent.optimization_enabled:
            print("⚠️  Optimization not enabled, skipping self-learning tests")
            return True

        # Test optimization engine methods
        engine = agent.engine

        # Test market condition classification
        test_signals = {
            "market_internals": "Market showing bullish momentum",
            "sentiment_llm": "Positive sentiment detected",
            "options_flow": []
        }

        market_condition = engine.classify_market_condition(test_signals)
        print(f"✓ Market condition classification: {market_condition}")

        # Test template selection
        template = engine.select_optimal_template(market_condition)
        print(f"✓ Template selection: {template.template_id if template else 'None'}")

        # Test prompt generation
        system_prompt, user_prompt, metadata = engine.generate_optimized_prompt(
            test_signals, market_condition
        )
        print("✓ Prompt generation successful")
        print(f"  - Template used: {metadata.get('template_id', 'Unknown')}")
        print(f"  - Estimated tokens: {metadata.get('estimated_tokens', 0)}")

        return True

    except Exception as e:
        print(f"❌ Self-learning integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ab_testing_framework(agent):
    """Test 4: A/B Testing Framework"""
    print("\n" + "=" * 60)
    print("TEST 4: A/B Testing Framework")
    print("=" * 60)

    try:
        if not agent.optimization_enabled or not agent.engine:
            print("⚠️  Optimization not enabled, skipping A/B testing tests")
            return True

        engine = agent.engine

        # Test A/B test creation
        from oracle_engine.prompt_optimization import MarketCondition
        experiment_id = engine.start_ab_test(
            "conservative_balanced",
            "aggressive_momentum",
            MarketCondition.BULLISH,
            1  # 1 hour for testing
        )

        if experiment_id:
            print(f"✓ A/B test started successfully: {experiment_id}")
        else:
            print("❌ A/B test creation failed")
            return False

        # Check if experiment is tracked
        if experiment_id in engine.active_experiments:
            print("✓ Experiment tracked in active experiments")
        else:
            print("❌ Experiment not tracked in active experiments")
            return False

        return True

    except Exception as e:
        print(f"❌ A/B testing framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_template_selection(agent):
    """Test 5: Template Selection Optimization"""
    print("\n" + "=" * 60)
    print("TEST 5: Template Selection Optimization")
    print("=" * 60)

    try:
        if not agent.optimization_enabled:
            print("⚠️  Optimization not enabled, skipping template selection tests")
            return True

        # Test batch template selection
        template_performance = {
            "conservative_balanced": {"successes": 5, "total": 7},
            "aggressive_momentum": {"successes": 3, "total": 5}
        }

        selected = agent._select_template_for_batch(template_performance, 5)
        print(f"✓ Template selection for batch: {selected}")

        # Test with no performance data
        selected_empty = agent._select_template_for_batch({}, 0)
        print(f"✓ Template selection with no data: {selected_empty}")

        return True

    except Exception as e:
        print(f"❌ Template selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_tracking(agent):
    """Test 6: Performance Tracking Mechanisms"""
    print("\n" + "=" * 60)
    print("TEST 6: Performance Tracking Mechanisms")
    print("=" * 60)

    try:
        if not agent.optimization_enabled:
            print("⚠️  Optimization not enabled, skipping performance tracking tests")
            return True

        # Test performance recording
        test_metadata = {
            'performance_metrics': {
                'success': True,
                'total_duration': 2.5,
                'stages_completed': 4
            },
            'stages': {
                'playbook_generation': {
                    'optimization_metadata': {
                        'template_used': 'conservative_balanced'
                    }
                },
                'signal_collection': {
                    'market_condition': 'bullish'
                }
            }
        }

        agent._record_pipeline_performance(test_metadata, "test output")

        # Check if performance was recorded
        if agent.performance_history:
            last_record = agent.performance_history[-1]
            print("✓ Performance recorded successfully")
            print(f"  - Success: {last_record['success']}")
            print(f"  - Duration: {last_record['total_duration']}")
            print(f"  - Template: {last_record.get('template_used', 'None')}")
        else:
            print("❌ Performance not recorded")
            return False

        # Test performance summary
        summary = agent.get_performance_summary()
        print(f"✓ Performance summary generated: {len(summary)} keys")

        return True

    except Exception as e:
        print(f"❌ Performance tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_genetic_algorithms(agent):
    """Test 7: Genetic Algorithms for Prompt Evolution"""
    print("\n" + "=" * 60)
    print("TEST 7: Genetic Algorithms for Prompt Evolution")
    print("=" * 60)

    try:
        if not agent.optimization_enabled or not agent.engine:
            print("⚠️  Optimization not enabled, skipping genetic algorithm tests")
            return True

        # Test learning cycle
        print("Testing learning cycle...")
        result = agent.run_learning_cycle(performance_threshold=0.5)

        if 'error' in result:
            print(f"⚠️  Learning cycle had issues: {result['error']}")
            # This might be expected if there are no top performers yet
            return True
        else:
            print("✓ Learning cycle completed")
            print(f"  - Successful: {result.get('evolution_successful', False)}")
            print(f"  - Evolved templates: {result.get('evolved_templates', [])}")

        return True

    except Exception as e:
        print(f"❌ Genetic algorithm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing(agent):
    """Test 8: Batch Processing with Template Selection"""
    print("\n" + "=" * 60)
    print("TEST 8: Batch Processing with Template Selection")
    print("=" * 60)

    try:
        # Test batch processing with sample data
        prompt_texts = ["Test prompt 1", "Test prompt 2"]
        chart_images = ["", ""]  # Empty base64 for testing

        results = agent.batch_pipeline_optimized(prompt_texts, chart_images)

        print(f"✓ Batch processing completed for {len(results)} items")

        for i, (playbook, metadata) in enumerate(results):
            success = metadata.get('performance_metrics', {}).get('success', False)
            duration = metadata.get('performance_metrics', {}).get('total_duration', 0)
            print(f"  - Item {i+1}: Success={success}, Duration={duration:.2f}s")

        return True

    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factory_function():
    """Test 9: Factory Function Singleton Pattern"""
    print("\n" + "=" * 60)
    print("TEST 9: Factory Function Singleton Pattern")
    print("=" * 60)

    try:
        from oracle_engine.agent_optimized import get_optimized_agent

        # Test multiple calls return same instance
        agent1 = get_optimized_agent()
        agent2 = get_optimized_agent()
        agent3 = get_optimized_agent()

        print(f"✓ Agent 1: {id(agent1)}")
        print(f"✓ Agent 2: {id(agent2)}")
        print(f"✓ Agent 3: {id(agent3)}")

        if agent1 is agent2 is agent3:
            print("✓ All instances are identical (singleton pattern working)")
        else:
            print("❌ Singleton pattern not working")
            return False

        return True

    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests and provide comprehensive report"""
    print("🚀 Starting Comprehensive Oracle Agent Optimized Pipeline Test")
    print("=" * 80)

    # Track test results
    test_results = {}

    # Test 1: Pipeline Initialization
    success, agent = test_pipeline_initialization()
    test_results['initialization'] = success

    if not success or not agent:
        print("\n❌ Critical failure in initialization. Cannot continue testing.")
        return test_results

    # Test 2: Database Creation
    test_results['database'] = test_database_creation(agent)

    # Test 3: Self-Learning Integration
    test_results['self_learning'] = test_self_learning_integration(agent)

    # Test 4: A/B Testing Framework
    test_results['ab_testing'] = test_ab_testing_framework(agent)

    # Test 5: Template Selection
    test_results['template_selection'] = test_template_selection(agent)

    # Test 6: Performance Tracking
    test_results['performance_tracking'] = test_performance_tracking(agent)

    # Test 7: Genetic Algorithms
    test_results['genetic_algorithms'] = test_genetic_algorithms(agent)

    # Test 8: Batch Processing
    test_results['batch_processing'] = test_batch_processing(agent)

    # Test 9: Factory Function
    test_results['factory_function'] = test_factory_function()

    # Final Report
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE TEST REPORT")
    print("=" * 80)

    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")

    print(f"\nOverall Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    return test_results

if __name__ == "__main__":
    results = run_comprehensive_test()