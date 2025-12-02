import pytest
from oracle_engine.agent import oracle_agent_batch_pipeline

# Dummy test data for batch pipeline
PROMPTS = [
    "TSLA, NVDA, SPY trending bullish after earnings beats. Options sweeps unusually high. Reddit sentiment surging.",
    "AAPL, MSFT, GOOG showing mixed signals. Options flow neutral. Awaiting earnings.",
]
CHARTS = [None, None]  # For now, no chart images


def test_batch_pipeline_runs():
    playbooks = oracle_agent_batch_pipeline(PROMPTS, CHARTS)
    assert isinstance(playbooks, list)
    assert len(playbooks) == len(PROMPTS)
    assert_playbooks_valid(playbooks)


# Helper for sub-assertions without a loop in the test
def assert_playbooks_valid(playbooks):
    assert all(isinstance(pb, str) for pb in playbooks)
    assert all(
        ("trades" in pb or "raw_output" in pb or "tomorrows_tape" in pb)
        for pb in playbooks
    )


if __name__ == "__main__":
    test_batch_pipeline_runs()
    print("Batch pipeline test passed.")
