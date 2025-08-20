import os
import pytest

from data_feeds.data_feed_orchestrator import (
    DataFeedOrchestrator,
    get_orchestrator,
    MarketBreadth,
    GroupPerformance,
)

@pytest.fixture(scope="module")
def orchestrator():
    return get_orchestrator()

@pytest.mark.parametrize("symbol", ["AAPL", "MSFT"])  # symbols should be broadly available
def test_quote(orchestrator, symbol):
    q = orchestrator.get_quote(symbol)
    if q is not None:
        assert q.symbol.upper() == symbol
        assert q.price is not None

@pytest.mark.parametrize("symbol,period,interval", [("AAPL", "1mo", "1d")])
def test_market_data(orchestrator, symbol, period, interval):
    md = orchestrator.get_market_data(symbol, period=period, interval=interval)
    if md is not None:
        assert md.symbol == symbol
        assert not md.data.empty

@pytest.mark.parametrize("symbol", ["AAPL"])        
def test_company_info(orchestrator, symbol):
    ci = orchestrator.get_company_info(symbol)
    if ci is not None:
        assert ci.symbol == symbol
        assert ci.name is not None

@pytest.mark.parametrize("symbol", ["AAPL"])        
def test_sentiment_map(orchestrator, symbol):
    smap = orchestrator.get_sentiment_data(symbol)
    # Map can be empty (no dependencies installed) but should be dict
    assert isinstance(smap, dict)
    for k,v in smap.items():
        assert hasattr(v, 'sentiment_score')

def test_market_breadth(orchestrator):
    mb = orchestrator.get_market_breadth()
    if mb is not None:
        assert isinstance(mb, MarketBreadth)

def test_sector_perf(orchestrator):
    sp = orchestrator.get_sector_performance()
    # list may be empty if FinViz fetch fails
    assert isinstance(sp, list)
    for item in sp:
        assert isinstance(item, GroupPerformance)

