def fetch_earnings_calendar(tickers=None) -> list:
    """
    Fetch upcoming earnings calendar data.
    TODO: Integrate with Finviz, Yahoo Calendar, or other provider.
    Returns:
        list: List of earnings events.
    """
    # Example placeholder data
    if tickers is None:
        tickers = ["AAPL", "TSLA"]
    return [
        {"ticker": ticker, "date": "2025-07-10", "estimate": 1.32} for ticker in tickers[:1]
    ] + [
        {"ticker": ticker, "date": "2025-07-12", "estimate": 2.15} for ticker in tickers[1:2]
    ]
