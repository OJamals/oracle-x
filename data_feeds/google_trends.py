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
    Returns a dict mapping keyword -> interest over time dict.
    Defensive against duplicate keywords and payload errors.
    """
    from pytrends.request import TrendReq
    import time

    # Normalize keywords to a unique list of strings
    if isinstance(keywords, str):
        kw_list = [keywords]
    else:
        try:
            kw_list = [str(k) for k in keywords]
        except Exception:
            kw_list = [str(keywords)]
    # Remove duplicates while preserving order
    seen = set()
    unique_kw = []
    for k in kw_list:
        if k not in seen:
            seen.add(k)
            unique_kw.append(k)
    # Avoid accidental character-splitting (e.g., list('AAPL')) by joining single-char lists back
    if len(unique_kw) > 1 and all(isinstance(k, str) and len(k) == 1 for k in unique_kw):
        unique_kw = ["".join(unique_kw)]

    all_data: dict = {}
    for i in range(0, len(unique_kw), batch_size):
        batch = unique_kw[i:i+batch_size]
        try:
            pytrends = TrendReq(hl='en-US', tz=0)
            pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
            data = pytrends.interest_over_time()
            if data is None:
                continue
            # Convert to dict and merge per-keyword
            if hasattr(data, "drop") and 'isPartial' in getattr(data, 'columns', []):
                data = data.drop(columns=['isPartial'])
            if hasattr(data, "to_dict"):
                # to_dict returns {col: {timestamp: value}}
                dict_data = data.to_dict()
                # Merge into all_data without overwriting existing keys unintentionally
                for k, series in dict_data.items():
                    if k not in all_data:
                        all_data[k] = series
                    else:
                        # Merge timestamps; prefer newer batch values
                        all_data[k].update(series)
            time.sleep(1)  # avoid rate limiting
        except Exception as e:
            print(f"[ERROR] Google Trends batch {batch} failed: {e}")
            continue
    return all_data
