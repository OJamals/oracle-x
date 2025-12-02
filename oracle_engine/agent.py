from openai import OpenAI
from oracle_engine.prompt_chain import (
    get_signals_from_scrapers,
    pull_similar_scenarios,
    adjust_scenario_tree_with_boost,
    batch_adjust_scenario_trees_with_boost,
    generate_final_playbook
)
import os
from core.config import config

API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = config.model.openai_api_base or os.environ.get("OPENAI_API_BASE", "https://api.githubcopilot.com/v1")
MODEL_NAME = config.model.openai_model

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

from typing import Optional

def oracle_agent_pipeline(prompt_text: str, chart_image_b64: Optional[str]) -> str:
    """
    Full Prompt Chain: real-time signals → ChromaDB recall → adjusted scenario tree → Playbook.
    Args:
        prompt_text (str): User prompt or market summary.
        chart_image_b64 (Optional[str]): Base64-encoded chart image.
    Returns:
        str: Final playbook JSON string.
    """
    # 1️⃣ Grab real-time signals from ALL scrapers
    signals = get_signals_from_scrapers(prompt_text, chart_image_b64 or "")
    print("[DEBUG] Signals sent to LLM:", signals)

    # 2️⃣ Pull similar historical scenarios from ChromaDB
    similar_scenarios = pull_similar_scenarios(prompt_text)
    print("[DEBUG] Similar scenarios from ChromaDB:", similar_scenarios)

    # 3️⃣ Adjust your scenario tree with all context (now using prompt boosting)
    scenario_tree = adjust_scenario_tree_with_boost(signals, similar_scenarios)
    print("[DEBUG] Scenario tree from LLM (boosted):", scenario_tree)

    # 4️⃣ Generate final Playbook (1–3 trades + Tomorrow's Tape)
    final_playbook = generate_final_playbook(signals, scenario_tree, model_name=MODEL_NAME)
    print("[DEBUG] Final playbook from LLM:", final_playbook)

    return final_playbook

def oracle_agent_batch_pipeline(prompt_texts: list, chart_image_b64s: list) -> list:
    """
    Batch pipeline: For a list of prompts and chart images, runs the full scenario tree and playbook generation pipeline in batch mode.
    Returns a list of playbook JSON strings.
    """
    signals_list = []
    similar_scenarios_list = []
    for prompt_text, chart_image_b64 in zip(prompt_texts, chart_image_b64s):
        signals = get_signals_from_scrapers(prompt_text, chart_image_b64)
        signals_list.append(signals)
        similar_scenarios = pull_similar_scenarios(prompt_text)
        similar_scenarios_list.append(similar_scenarios)
    scenario_trees = batch_adjust_scenario_trees_with_boost(signals_list, similar_scenarios_list)
    playbooks = []
    for signals, scenario_tree in zip(signals_list, scenario_trees):
        playbook = generate_final_playbook(signals, scenario_tree, model_name=MODEL_NAME)
        playbooks.append(playbook)
    return playbooks
