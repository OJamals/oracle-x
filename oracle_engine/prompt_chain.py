"""
Compatibility wrapper for prompt chain API.

The prompt chain implementation now lives under oracle_engine.chains.prompt_chain.
This module re-exports the public surface to maintain existing imports.
"""

from oracle_engine.chains.prompt_chain import *  # noqa: F401,F403
