import os

from data_feeds.data_feed_orchestrator import (
    get_orchestrator,
    get_quote as orch_get_quote,
    get_market_data as orch_get_market_data,
    get_market_breadth,
    DataSource,
)


def main():
    orch = get_orchestrator()

    symbol = "AAPL"
    api_key = os.getenv("TWELVEDATA_API_KEY")

    # Skip Twelve Data paths if API key is missing
    if not api_key:
        print("Skipping Twelve Data tests; TWELVEDATA_API_KEY not set")

    print("== Smoke: Twelve Data Quote ==")
    if api_key:
        q = orch_get_quote(symbol)
        q = orch.get_quote(symbol, preferred_sources=[DataSource.TWELVE_DATA])
        if q:
            print(
                f"{symbol} price={q.price} change={q.change} change%={q.change_percent} ts={q.timestamp} src={q.source}"
            )
        else:
            print("Quote unavailable")
    else:
        print("TWELVEDATA_API_KEY not set, skipping quote")

    print("\n== Smoke: Twelve Data Market Data (daily) ==")
    if api_key:
        md = orch.get_market_data(
            symbol,
            period="1y",
            interval="1d",
            preferred_sources=[DataSource.TWELVE_DATA],
        )
        if md and md.data is not None and not md.data.empty:
            print(f"{symbol} last 3 bars:")
            print(md.data.tail(3))
        else:
            print("Market data unavailable")
    else:
        print("TWELVEDATA_API_KEY not set, skipping market data")

    print("\n== Smoke: FinViz Market Breadth ==")
    breadth = get_market_breadth()
    if breadth:
        print(
            f"Advancers={breadth.advancers} Decliners={breadth.decliners} AsOf={breadth.as_of} Source={breadth.source}"
        )
    else:
        print("Market breadth unavailable")


if __name__ == "__main__":
    main()
