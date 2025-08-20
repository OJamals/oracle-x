"""
Enhanced Prompt Chain with Self-Learning Optimization

This module extends the existing prompt_chain.py with advanced optimization capabilities,
integrating the PromptOptimizationEngine for intelligent prompt generation and 
performance-based learning.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from oracle_engine.prompt_optimization import (
    get_optimization_engine, 
    MarketCondition, 
    PromptStrategy
)
from oracle_engine.prompt_chain import (
    get_signals_from_scrapers as _original_get_signals,
    pull_similar_scenarios as _original_pull_scenarios,
    clean_signals_for_llm,
    extract_scenario_tree,
    MODEL_NAME
)
from oracle_engine.model_attempt_logger import log_attempt, pop_attempts
from openai import OpenAI
import os
import config_manager
import time

logger = logging.getLogger(__name__)

# Initialize OpenAI client
API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = config_manager.get_openai_api_base() or os.environ.get("OPENAI_API_BASE", "https://api.githubcopilot.com/v1")
client = OpenAI(api_key=API_KEY, base_url=API_BASE)

def get_signals_from_scrapers_optimized(prompt_text: str, chart_image_b64: str) -> Dict[str, Any]:
    """
    Enhanced version of get_signals_from_scrapers with intelligent preprocessing
    """
    # Get raw signals using original function
    raw_signals = _original_get_signals(prompt_text, chart_image_b64)
    
    # Get optimization engine
    engine = get_optimization_engine()
    
    # Classify market condition
    market_condition = engine.classify_market_condition(raw_signals)
    logger.info(f"Detected market condition: {market_condition.value}")
    
    # Add market condition metadata
    raw_signals['_market_condition'] = market_condition.value
    raw_signals['_optimization_metadata'] = {
        'classification_timestamp': time.time(),
        'signal_count': len(raw_signals),
        'prompt_text_length': len(prompt_text)
    }
    
    return raw_signals

def adjust_scenario_tree_optimized(
    signals: Dict[str, Any], 
    similar_scenarios: str, 
    model_name: str = MODEL_NAME,
    template_id: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced scenario tree adjustment with optimized prompts and performance tracking
    """
    engine = get_optimization_engine()
    
    # Extract market condition from signals metadata
    market_condition_str = signals.get('_market_condition', 'sideways')
    try:
        market_condition = MarketCondition(market_condition_str)
    except ValueError:
        market_condition = MarketCondition.SIDEWAYS
    
    # Generate optimized prompt
    system_prompt, user_prompt, prompt_metadata = engine.generate_optimized_prompt(
        signals, market_condition, template_id
    )
    
    logger.info(f"Using template: {prompt_metadata['template_id']} for {market_condition.value} market")
    
    # Execute LLM call with optimized prompt
    start_time = time.time()
    attempts = []
    
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=600,
            temperature=0.3
        )
        
        content = (resp.choices[0].message.content or "").strip()
        
        if content:
            log_attempt("adjust_scenario_tree_optimized", model_name, 
                       start_time=start_time, success=True, empty=False, error=None)
            attempts.append({
                'purpose': 'adjust_scenario_tree_optimized',
                'model': model_name,
                'success': True,
                'empty': False,
                'error': None,
                'latency_sec': time.time() - start_time
            })
        else:
            log_attempt("adjust_scenario_tree_optimized", model_name,
                       start_time=start_time, success=False, empty=True, error=None)
            attempts.append({
                'purpose': 'adjust_scenario_tree_optimized',
                'model': model_name,
                'success': False,
                'empty': True,
                'error': None,
                'latency_sec': time.time() - start_time
            })
            content = ""
            
    except Exception as e:
        log_attempt("adjust_scenario_tree_optimized", model_name,
                   start_time=start_time, success=False, empty=False, error=str(e))
        attempts.append({
            'purpose': 'adjust_scenario_tree_optimized',
            'model': model_name,
            'success': False,
            'empty': False,
            'error': str(e),
            'latency_sec': time.time() - start_time
        })
        content = ""
        logger.error(f"Model call failed: {e}")
    
    # Record performance for optimization
    engine.record_prompt_performance(prompt_metadata, attempts)
    
    # Add optimization metadata to response
    optimization_metadata = {
        'prompt_metadata': prompt_metadata,
        'performance_data': attempts,
        'market_condition': market_condition.value
    }
    
    return content, optimization_metadata

def generate_final_playbook_optimized(
    signals: Dict[str, Any], 
    scenario_tree: str, 
    model_name: str = MODEL_NAME,
    template_id: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced playbook generation with optimized prompts and comprehensive tracking
    """
    engine = get_optimization_engine()
    
    # Clean signals first
    cleaned_signals = clean_signals_for_llm(signals)
    
    # Extract market condition
    market_condition_str = signals.get('_market_condition', 'sideways')
    try:
        market_condition = MarketCondition(market_condition_str)
    except ValueError:
        market_condition = MarketCondition.SIDEWAYS
    
    # For playbook generation, prefer templates that focus on trading execution
    if template_id is None:
        # Select template optimized for final trading decisions
        template_preferences = [
            ('conservative_balanced', MarketCondition.SIDEWAYS),
            ('aggressive_momentum', MarketCondition.BULLISH),
            ('aggressive_momentum', MarketCondition.VOLATILE),
            ('earnings_specialist', MarketCondition.EARNINGS),
            ('technical_precision', MarketCondition.BEARISH)
        ]
        
        for temp_id, condition in template_preferences:
            if condition == market_condition:
                template_id = temp_id
                break
        
        if template_id is None:
            template_id = 'conservative_balanced'  # Safe default
    
    # Generate optimized prompt for playbook generation
    system_prompt, user_prompt_base, prompt_metadata = engine.generate_optimized_prompt(
        cleaned_signals, market_condition, template_id
    )
    
    # Enhance prompt with scenario tree context
    enhanced_user_prompt = f"""
{user_prompt_base}

Adjusted Scenario Tree:
{scenario_tree}

Based on this analysis, generate the 1–3 highest-confidence trades for tomorrow.
Include: ticker, direction, instrument, entry range, profit target, stop-loss,
counter-signal, and a 5-sentence 'Tomorrow's Tape'.

Format your response as **valid JSON only** (no markdown, no extra text, no comments) with:
- 'trades': a list of trade objects
- 'tomorrows_tape': a string summary

IMPORTANT: For each trade, include a 'thesis' field summarizing the core rationale in 1-2 sentences.
IMPORTANT: For each trade, include a 'scenario_tree' field with a dictionary of scenario probabilities (base_case, bull_case, bear_case), e.g.:
"scenario_tree": {{"base_case": "70% - ...", "bull_case": "20% - ...", "bear_case": "10% - ..."}}

DO NOT include any text before or after the JSON. Output only the JSON object.
"""
    
    # Execute LLM call
    start_time = time.time()
    attempts = []
    
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_user_prompt}
            ],
            max_completion_tokens=1024,
            temperature=0.3
        )
        
        content = (resp.choices[0].message.content or "").strip()
        
        if content:
            log_attempt("generate_final_playbook_optimized", model_name,
                       start_time=start_time, success=True, empty=False, error=None)
            attempts.append({
                'purpose': 'generate_final_playbook_optimized',
                'model': model_name,
                'success': True,
                'empty': False,
                'error': None,
                'latency_sec': time.time() - start_time
            })
        else:
            log_attempt("generate_final_playbook_optimized", model_name,
                       start_time=start_time, success=False, empty=True, error=None)
            attempts.append({
                'purpose': 'generate_final_playbook_optimized',
                'model': model_name,
                'success': False,
                'empty': True,
                'error': None,
                'latency_sec': time.time() - start_time
            })
            content = ""
            
    except Exception as e:
        log_attempt("generate_final_playbook_optimized", model_name,
                   start_time=start_time, success=False, empty=False, error=str(e))
        attempts.append({
            'purpose': 'generate_final_playbook_optimized',
            'model': model_name,
            'success': False,
            'empty': False,
            'error': str(e),
            'latency_sec': time.time() - start_time
        })
        content = ""
        logger.error(f"Final playbook generation failed: {e}")
    
    # Record performance
    engine.record_prompt_performance(prompt_metadata, attempts)
    
    # Analyze output quality
    output_quality = analyze_playbook_quality(content)
    
    optimization_metadata = {
        'prompt_metadata': prompt_metadata,
        'performance_data': attempts,
        'market_condition': market_condition.value,
        'output_quality': output_quality,
        'template_used': template_id
    }
    
    return content, optimization_metadata

def analyze_playbook_quality(playbook_json: str) -> Dict[str, Any]:
    """
    Analyze the quality of generated playbook for learning purposes
    """
    quality_metrics = {
        'valid_json': False,
        'has_trades': False,
        'has_tomorrows_tape': False,
        'trade_count': 0,
        'trades_have_required_fields': False,
        'confidence_score': 0.0,
        'completeness_score': 0.0
    }
    
    try:
        # Parse JSON
        data = json.loads(playbook_json)
        quality_metrics['valid_json'] = True
        
        # Check structure
        if 'trades' in data and isinstance(data['trades'], list):
            quality_metrics['has_trades'] = True
            quality_metrics['trade_count'] = len(data['trades'])
            
            # Check trade fields
            required_fields = ['ticker', 'direction', 'entry_range', 'profit_target', 'stop_loss']
            if data['trades']:
                first_trade = data['trades'][0]
                has_required = all(field in first_trade for field in required_fields)
                quality_metrics['trades_have_required_fields'] = has_required
        
        if 'tomorrows_tape' in data and isinstance(data['tomorrows_tape'], str) and data['tomorrows_tape'].strip():
            quality_metrics['has_tomorrows_tape'] = True
        
        # Calculate overall scores
        structure_score = sum([
            quality_metrics['valid_json'],
            quality_metrics['has_trades'],
            quality_metrics['has_tomorrows_tape'],
            quality_metrics['trades_have_required_fields']
        ]) / 4
        
        content_score = min(1.0, quality_metrics['trade_count'] / 2)  # Optimal is 1-2 trades
        
        quality_metrics['completeness_score'] = (structure_score + content_score) / 2
        quality_metrics['confidence_score'] = quality_metrics['completeness_score']
        
    except json.JSONDecodeError:
        quality_metrics['valid_json'] = False
    except Exception as e:
        logger.warning(f"Error analyzing playbook quality: {e}")
    
    return quality_metrics

def run_optimization_experiment(signals: Dict[str, Any], duration_minutes: int = 30) -> Dict[str, Any]:
    """
    Run an optimization experiment comparing different prompt strategies
    """
    engine = get_optimization_engine()
    market_condition = engine.classify_market_condition(signals)
    
    # Select templates to test
    test_templates = ['conservative_balanced', 'aggressive_momentum', 'technical_precision']
    available_templates = [tid for tid in test_templates if tid in engine.prompt_templates]
    
    if len(available_templates) < 2:
        logger.warning("Not enough templates available for experiment")
        return {}
    
    # Run A/B test
    template_a, template_b = available_templates[0], available_templates[1]
    experiment_id = engine.start_ab_test(template_a, template_b, market_condition, duration_minutes // 60)
    
    logger.info(f"Started optimization experiment {experiment_id}: {template_a} vs {template_b}")
    
    results = {}
    for template_id in [template_a, template_b]:
        system_prompt, user_prompt, metadata = engine.generate_optimized_prompt(signals, market_condition, template_id)
        
        # Generate sample outputs
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=600
            )
            
            content = resp.choices[0].message.content or ""
            quality = analyze_playbook_quality(content)
            
            results[template_id] = {
                'metadata': metadata,
                'output_quality': quality,
                'content_length': len(content),
                'success': len(content) > 0
            }
            
        except Exception as e:
            results[template_id] = {
                'metadata': metadata,
                'error': str(e),
                'success': False
            }
    
    return {
        'experiment_id': experiment_id,
        'market_condition': market_condition.value,
        'results': results,
        'recommendation': _analyze_experiment_results(results)
    }

def _analyze_experiment_results(results: Dict[str, Any]) -> str:
    """Analyze experiment results and provide recommendation"""
    if len(results) < 2:
        return "Insufficient data for analysis"
    
    template_scores = {}
    for template_id, result in results.items():
        if result.get('success'):
            quality = result.get('output_quality', {})
            score = quality.get('completeness_score', 0)
            template_scores[template_id] = score
        else:
            template_scores[template_id] = 0
    
    if not template_scores:
        return "All templates failed"
    
    best_template = max(template_scores.items(), key=lambda x: x[1])
    return f"Recommended template: {best_template[0]} (score: {best_template[1]:.2f})"

def get_optimization_analytics() -> Dict[str, Any]:
    """Get comprehensive optimization analytics"""
    engine = get_optimization_engine()
    return engine.get_performance_analytics()

def evolve_prompt_templates(performance_threshold: float = 0.7) -> List[str]:
    """Evolve prompt templates based on performance data"""
    engine = get_optimization_engine()
    analytics = engine.get_performance_analytics()
    
    # Identify top performers
    top_performers = []
    for template_perf in analytics.get('template_performance', []):
        if template_perf['avg_success_rate'] >= performance_threshold:
            top_performers.append(template_perf['template_id'])
    
    if not top_performers:
        logger.warning("No top performing templates found for evolution")
        return []
    
    # Evolve templates
    new_templates = engine.evolve_prompts(top_performers)
    evolved_ids = []
    
    for template in new_templates:
        engine.prompt_templates[template.template_id] = template
        evolved_ids.append(template.template_id)
        logger.info(f"Created evolved template: {template.template_id}")
    
    return evolved_ids
