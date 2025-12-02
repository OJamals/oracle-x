"""
Stub for options_flow to fix test imports during refactor.
"""

def fetch_options_flow(symbols: list[str]) -> dict:
    """Stub implementation for options flow data."""
    return {
        "data_source": "yfinance_options_stub",
        "total_sweeps": len(symbols) * 2,
        "unusual_sweeps": [
            {
                "ticker": symbols[0] if symbols else "AAPL",
                "direction": "call",
                "strike": 150.0,
                "volume": 1000,
                "premium": 2.5,
            }
            for _ in range(3)
        ],
    }