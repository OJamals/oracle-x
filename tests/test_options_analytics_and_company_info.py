import pytest
from data_feeds.data_feed_orchestrator import get_orchestrator

orchestrator = get_orchestrator()

def test_options_analytics():
    result = orchestrator.get_options_analytics("AAPL")
    assert isinstance(result, dict)
    # Should contain keys for chain, iv, greeks, gex, max_pain
    for key in ["chain", "iv", "greeks", "gex", "max_pain"]:
        assert key in result or key in result.keys()

def test_company_info():
    info = orchestrator.get_company_info("AAPL")
    assert info is None or hasattr(info, "symbol")
    if info:
        assert info.symbol == "AAPL"
        assert info.name is not None
