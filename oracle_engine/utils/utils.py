"""
Shared utilities for oracle_engine modules.

This module contains common helper functions, prompt builders, and chain validators
used across the oracle engine components.
"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def build_prompt_template(base_template: str, variables: Dict[str, Any]) -> str:
    """Build a prompt from template with variable substitution."""
    try:
        return base_template.format(**variables)
    except KeyError as e:
        logger.warning(f"Missing variable in prompt template: {e}")
        return base_template


def validate_chain_output(output: str, required_fields: List[str]) -> bool:
    """Validate that chain output contains required fields."""
    if not output:
        return False
    
    # Basic validation - can be enhanced based on output format
    for field in required_fields:
        if field not in output:
            return False
    return True


def sanitize_prompt_input(text: str, max_length: int = 10000) -> str:
    """Sanitize and truncate prompt input."""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response, handling various formats."""
    import json
    import re
    
    # Try direct JSON parse
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract from markdown code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON-like content
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def calculate_confidence_score(output: str, criteria: List[str]) -> float:
    """Calculate confidence score based on presence of criteria in output."""
    if not output:
        return 0.0
    
    score = 0.0
    for criterion in criteria:
        if criterion.lower() in output.lower():
            score += 1.0
    
    return min(score / len(criteria), 1.0) if criteria else 0.0

import re
from collections.abc import Sequence
from typing import Dict, Any

def clean_signals_for_llm(signals