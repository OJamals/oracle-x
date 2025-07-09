def fetch_options_flow(tickers=None) -> dict:
    """
    Fetch unusual options flow data.
    TODO: Integrate with Unusual Whales, FlowAlgo, or other provider.
    Returns:
        dict: Unusual options sweeps.
    """
    # Example placeholder data
    if tickers is None:
        tickers = ["AAPL", "TSLA"]
    unusual_sweeps = [
        {"ticker": ticker, "direction": "Call", "strike": 200, "volume": 10000} for ticker in tickers[:1]
    ] + [
        {"ticker": ticker, "direction": "Put", "strike": 750, "volume": 8000} for ticker in tickers[1:2]
    ]
    return {"unusual_sweeps": unusual_sweeps}
