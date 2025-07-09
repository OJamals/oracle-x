from typing import List
from .qdrant_store import query_similar, batch_query_similar


def build_boosted_prompt(base_prompt: str, trade_thesis: str) -> str:
    """
    Takes your base prompt + thesis for a trade idea,
    queries Qdrant for similar past scenarios,
    and appends the top matches to your LLM prompt.
    """
    hits = query_similar(trade_thesis, top_k=3)

    if not hits:
        print("No similar scenarios found in Qdrant.")
        return base_prompt

    boost_text = "\n\n### SIMILAR PAST SCENARIOS:\n"
    for hit in hits:
        payload = hit.payload
        boost_text += (
            f"- Ticker: {payload['ticker']}, "
            f"Direction: {payload['direction']}, "
            f"Thesis: {payload['thesis']}, "
            f"Date: {payload['date']}\n"
        )

    print(f"✅ Found {len(hits)} similar scenarios. Boosting prompt.")
    return f"{base_prompt}\n\n{boost_text}"


def batch_build_boosted_prompts(base_prompts: List[str], trade_theses: List[str]) -> List[str]:
    """
    Batch version of build_boosted_prompt for multiple prompts/theses.
    Returns a list of boosted prompts.
    """
    all_hits = batch_query_similar(trade_theses, top_k=3)
    boosted = []
    for base_prompt, hits in zip(base_prompts, all_hits):
        if not hits:
            boosted.append(base_prompt)
            continue
        boost_text = "\n\n### SIMILAR PAST SCENARIOS:\n"
        for hit in hits:
            payload = hit.payload
            boost_text += (
                f"- Ticker: {payload['ticker']}, "
                f"Direction: {payload['direction']}, "
                f"Thesis: {payload['thesis']}, "
                f"Date: {payload['date']}\n"
            )
        boosted.append(f"{base_prompt}\n\n{boost_text}")
    return boosted
