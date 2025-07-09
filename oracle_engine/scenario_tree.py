def build_scenario_tree(trade_idea):
    print(f"Building scenario tree for {trade_idea['ticker']}...")
    return {
        "base_case": "70% - Strong follow-through on unusual options activity",
        "bull_case": "20% - Macro news surprise enhances move",
        "bear_case": "10% - Unexpected reversal, watch overnight futures"
    }
