import warnings
warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated*",
    category=FutureWarning
)
def fetch_google_trends(keywords, timeframe="now 7-d", geo="US", batch_size=5) -> dict:
    """
    Fetch Google Trends data for a list of keywords using pytrends (free, open-source).
    Batches requests to avoid Google 400 errors (max 5 keywords per batch).
    Returns a dict of interest over time for each keyword.
    """
    from pytrends.request import TrendReq
    import time
    all_data = {}
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i+batch_size]
        try:
            pytrends = TrendReq()
            pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
            data = pytrends.interest_over_time()
            if not data.empty:
                all_data |= data.to_dict()
            time.sleep(1)  # avoid rate limiting
        except Exception as e:
            print(f"[ERROR] Google Trends batch {batch} failed: {e}")
            continue
    return all_data
