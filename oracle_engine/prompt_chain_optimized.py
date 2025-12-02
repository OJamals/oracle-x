"""
Compatibility wrapper for optimized prompt chain helpers.

The optimized functionality now lives in oracle_engine.chains.prompt_chain;
this module re-exports the public API to keep existing imports working.
"""

from oracle_engine.chains.prompt_chain import (
    adjust_scenario_tree_optimized,
    analyze_playbook_quality,
    evolve_prompt_templates,
    generate_final_playbook_optimized,
    get_optimization_analytics,
    get_signals_from_scrapers_optimized,
    run_optimization_experiment,
)

__all__ = [
    "adjust_scenario_tree_optimized",
    "analyze_playbook_quality",
    "evolve_prompt_templates",
    "generate_final_playbook_optimized",
    "get_optimization_analytics",
    "get_signals_from_scrapers_optimized",
    "run_optimization_experiment",
]
