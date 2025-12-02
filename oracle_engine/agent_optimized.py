"""
Enhanced Oracle Agent with Self-Learning Prompt Optimization

This module extends the existing agent.py with advanced optimization capabilities,
providing intelligent prompt selection, performance tracking, and continuous learning.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from oracle_engine.prompt_chain import (
    get_signals_from_scrapers,
    adjust_scenario_tree,
    generate_final_playbook,
    run_optimization_experiment,
    get_optimization_analytics,
    evolve_prompt_templates
)
from oracle_engine.prompt_chain import pull_similar_scenarios
from oracle_engine.prompt_optimization import get_optimization_engine, MarketCondition
from core.config import config
import time
import json

logger = logging.getLogger(__name__)

class OracleAgentOptimized:
    """
    Enhanced Oracle Agent with self-learning prompt optimization
    """
    
    def __init__(self, optimization_enabled: bool = True):
        self.optimization_enabled = optimization_enabled
        self.engine = get_optimization_engine() if optimization_enabled else None
        self.performance_history = []
        self.experiment_queue = []
        
    def oracle_agent_pipeline_optimized(
        self, 
        prompt_text: str, 
        chart_image_b64: Optional[str],
        template_override: Optional[str] = None,
        enable_experiments: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced Oracle pipeline with optimization and performance tracking
        
        Args:
            prompt_text: User prompt or market summary
            chart_image_b64: Base64-encoded chart image
            template_override: Optional specific template to use
            enable_experiments: Whether to run optimization experiments
            
        Returns:
            Tuple of (final playbook JSON, optimization metadata)
        """
        pipeline_start = time.time()
        metadata = {
            'pipeline_version': 'optimized_v1.0',
            'optimization_enabled': self.optimization_enabled,
            'stages': {},
            'performance_metrics': {}
        }
        
        try:
            # Stage 1: Enhanced signal collection
            logger.info("🔍 Stage 1: Enhanced signal collection")
            stage_start = time.time()
            
            if self.optimization_enabled:
                signals = get_signals_from_scrapers(prompt_text, chart_image_b64 or "", optimize=True)
            else:
                signals = get_signals_from_scrapers(prompt_text, chart_image_b64 or "")
            
            metadata['stages']['signal_collection'] = {
                'duration': time.time() - stage_start,
                'signal_count': len(signals),
                'market_condition': signals.get('_market_condition', 'unknown')
            }
            
            logger.info(f"✅ Collected {len(signals)} signals in {metadata['stages']['signal_collection']['duration']:.2f}s")
            
            # Stage 2: Historical scenario retrieval
            logger.info("🔍 Stage 2: Historical scenario retrieval")
            stage_start = time.time()
            
            similar_scenarios = pull_similar_scenarios(prompt_text)
            
            metadata['stages']['scenario_retrieval'] = {
                'duration': time.time() - stage_start,
                'scenarios_found': len(similar_scenarios.split('\n')) if similar_scenarios else 0
            }
            
            logger.info(f"✅ Retrieved scenarios in {metadata['stages']['scenario_retrieval']['duration']:.2f}s")
            
            # Stage 3: Optimized scenario tree generation
            logger.info("🔍 Stage 3: Optimized scenario tree generation")
            stage_start = time.time()
            
            if self.optimization_enabled:
                scenario_tree = adjust_scenario_tree(
                    signals, similar_scenarios, optimize=True, template_id=template_override
                )
                scenario_metadata = {}  # For compatibility
            else:
                scenario_tree = adjust_scenario_tree(signals, similar_scenarios)
                scenario_metadata = {}
            
            metadata['stages']['scenario_tree'] = {
                'duration': time.time() - stage_start,
                'optimization_metadata': scenario_metadata,
                'content_length': len(scenario_tree)
            }
            
            logger.info(f"✅ Generated scenario tree in {metadata['stages']['scenario_tree']['duration']:.2f}s")
            
            # Stage 4: Optimized playbook generation
            logger.info("🔍 Stage 4: Optimized playbook generation")
            stage_start = time.time()
            
            if self.optimization_enabled:
                final_playbook = generate_final_playbook(
                    signals, scenario_tree, optimize=True, template_id=template_override
                )
                playbook_metadata = {}  # For compatibility
            else:
                final_playbook = generate_final_playbook(signals, scenario_tree)
                playbook_metadata = {}
            
            metadata['stages']['playbook_generation'] = {
                'duration': time.time() - stage_start,
                'optimization_metadata': playbook_metadata,
                'content_length': len(final_playbook)
            }
            
            logger.info(f"✅ Generated playbook in {metadata['stages']['playbook_generation']['duration']:.2f}s")
            
            # Stage 5: Optional optimization experiments
            if enable_experiments and self.optimization_enabled:
                logger.info("🔍 Stage 5: Running optimization experiments")
                stage_start = time.time()
                
                experiment_results = run_optimization_experiment(signals)
                
                metadata['stages']['experiments'] = {
                    'duration': time.time() - stage_start,
                    'experiment_results': experiment_results
                }
                
                logger.info(f"✅ Completed experiments in {metadata['stages']['experiments']['duration']:.2f}s")
            
            # Calculate overall performance metrics
            total_duration = time.time() - pipeline_start
            metadata['performance_metrics'] = {
                'total_duration': total_duration,
                'stages_completed': len(metadata['stages']),
                'success': len(final_playbook) > 0,
                'tokens_estimated': sum(
                    stage.get('content_length', 0) 
                    for stage in metadata['stages'].values()
                ) // 4
            }
            
            # Store performance for learning
            if self.optimization_enabled:
                self._record_pipeline_performance(metadata, final_playbook)
            
            logger.info(f"🎯 Pipeline completed successfully in {total_duration:.2f}s")
            return final_playbook, metadata
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            metadata['error'] = str(e)
            metadata['performance_metrics'] = {
                'total_duration': time.time() - pipeline_start,
                'success': False
            }
            return "", metadata
    
    def batch_pipeline_optimized(
        self, 
        prompt_texts: List[str], 
        chart_image_b64s: List[str],
        enable_optimization: bool = True
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Optimized batch processing with intelligent template selection
        """
        results = []
        template_performance = {}
        
        logger.info(f"🚀 Starting optimized batch pipeline for {len(prompt_texts)} items")
        
        for i, (prompt_text, chart_image) in enumerate(zip(prompt_texts, chart_image_b64s)):
            logger.info(f"📊 Processing batch item {i+1}/{len(prompt_texts)}")
            
            # Intelligent template selection based on previous performance
            template_id = self._select_template_for_batch(template_performance, i)
            
            playbook, metadata = self.oracle_agent_pipeline_optimized(
                prompt_text, chart_image, template_override=template_id
            )
            
            # Update template performance tracking
            if metadata.get('performance_metrics', {}).get('success'):
                template_used = metadata.get('stages', {}).get('playbook_generation', {}).get('optimization_metadata', {}).get('template_used')
                if template_used:
                    if template_used not in template_performance:
                        template_performance[template_used] = {'successes': 0, 'total': 0}
                    template_performance[template_used]['successes'] += 1
                    template_performance[template_used]['total'] += 1
            
            results.append((playbook, metadata))
        
        logger.info(f"✅ Batch processing completed. Template performance: {template_performance}")
        return results
    
    def _select_template_for_batch(self, template_performance: Dict, batch_index: int) -> Optional[str]:
        """Select optimal template for batch processing based on performance"""
        if not self.optimization_enabled or not template_performance:
            return None
        
        # For first few items, use different templates to gather data
        if batch_index < 3:
            templates = ['conservative_balanced', 'aggressive_momentum', 'technical_precision']
            return templates[batch_index % len(templates)]
        
        # For later items, use best performing template
        best_template = None
        best_rate = 0
        
        for template, stats in template_performance.items():
            if stats['total'] > 0:
                success_rate = stats['successes'] / stats['total']
                if success_rate > best_rate:
                    best_rate = success_rate
                    best_template = template
        
        return best_template
    
    def _record_pipeline_performance(self, metadata: Dict[str, Any], output: str):
        """Record pipeline performance for learning"""
        performance_record = {
            'timestamp': time.time(),
            'success': metadata['performance_metrics']['success'],
            'total_duration': metadata['performance_metrics']['total_duration'],
            'output_length': len(output),
            'stages_completed': metadata['performance_metrics']['stages_completed'],
            'market_condition': metadata.get('stages', {}).get('signal_collection', {}).get('market_condition'),
            'template_used': metadata.get('stages', {}).get('playbook_generation', {}).get('optimization_metadata', {}).get('template_used')
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for recent operations"""
        if not self.optimization_enabled:
            return {'error': 'Optimization not enabled'}
        
        # Get optimization analytics
        analytics = get_optimization_analytics()
        
        # Calculate pipeline-specific metrics
        recent_history = [
            record for record in self.performance_history
            if time.time() - record['timestamp'] <= days * 24 * 3600
        ]
        
        if recent_history:
            success_rate = sum(1 for r in recent_history if r['success']) / len(recent_history)
            avg_duration = sum(r['total_duration'] for r in recent_history) / len(recent_history)
            
            # Template performance
            template_stats = {}
            for record in recent_history:
                template = record.get('template_used')
                if template:
                    if template not in template_stats:
                        template_stats[template] = {'count': 0, 'successes': 0}
                    template_stats[template]['count'] += 1
                    if record['success']:
                        template_stats[template]['successes'] += 1
            
            pipeline_metrics = {
                'total_runs': len(recent_history),
                'success_rate': success_rate,
                'avg_duration_seconds': avg_duration,
                'template_performance': {
                    template: {
                        'usage_count': stats['count'],
                        'success_rate': stats['successes'] / stats['count'] if stats['count'] > 0 else 0
                    }
                    for template, stats in template_stats.items()
                }
            }
        else:
            pipeline_metrics = {'total_runs': 0, 'message': 'No recent data available'}
        
        return {
            'pipeline_metrics': pipeline_metrics,
            'optimization_analytics': analytics,
            'optimization_enabled': self.optimization_enabled
        }
    
    def run_learning_cycle(self, performance_threshold: float = 0.7) -> Dict[str, Any]:
        """Run a learning cycle to evolve prompt templates"""
        if not self.optimization_enabled:
            return {'error': 'Optimization not enabled'}
        
        logger.info("🧠 Starting learning cycle for prompt evolution")
        
        try:
            # Evolve templates based on performance
            evolved_templates = evolve_prompt_templates(performance_threshold)
            
            # Get updated analytics
            analytics = get_optimization_analytics()
            
            result = {
                'evolved_templates': evolved_templates,
                'evolution_successful': len(evolved_templates) > 0,
                'performance_threshold': performance_threshold,
                'analytics_summary': {
                    'total_templates': analytics.get('total_templates', 0),
                    'active_experiments': analytics.get('active_experiments', 0)
                }
            }
            
            logger.info(f"✅ Learning cycle completed. Evolved {len(evolved_templates)} templates")
            return result
            
        except Exception as e:
            logger.error(f"❌ Learning cycle failed: {e}")
            return {'error': str(e), 'evolution_successful': False}
    
    def start_optimization_experiment(
        self, 
        template_a: str, 
        template_b: str, 
        market_condition: MarketCondition,
        duration_hours: int = 24
    ) -> str:
        """Start an A/B test between two templates"""
        if not self.optimization_enabled or self.engine is None:
            return "Optimization not enabled"
        
        experiment_id = self.engine.start_ab_test(template_a, template_b, market_condition, duration_hours)
        logger.info(f"🧪 Started A/B test {experiment_id}: {template_a} vs {template_b}")
        return experiment_id
    
    def enable_adaptive_learning(self, enable: bool = True):
        """Enable or disable adaptive learning features"""
        self.optimization_enabled = enable
        if enable and self.engine is None:
            self.engine = get_optimization_engine()
        logger.info(f"🔧 Adaptive learning {'enabled' if enable else 'disabled'}")

# Create global instance
_optimized_agent = None

def get_optimized_agent() -> OracleAgentOptimized:
    """Get global optimized agent instance"""
    global _optimized_agent
    if _optimized_agent is None:
        _optimized_agent = OracleAgentOptimized()
    return _optimized_agent

# Backward compatibility functions
def oracle_agent_pipeline_optimized(prompt_text: str, chart_image_b64: Optional[str]) -> str:
    """Backward compatible interface to optimized pipeline"""
    agent = get_optimized_agent()
    playbook, metadata = agent.oracle_agent_pipeline_optimized(prompt_text, chart_image_b64)
    return playbook

def oracle_agent_batch_pipeline_optimized(prompt_texts: List[str], chart_image_b64s: List[str]) -> List[str]:
    """Backward compatible interface to optimized batch pipeline"""
    agent = get_optimized_agent()
    results = agent.batch_pipeline_optimized(prompt_texts, chart_image_b64s)
    return [playbook for playbook, metadata in results]
