
"""Reddit sentiment extraction with optional concurrency and early-exit heuristics."""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Tuple, Set, Any

try:  # Optional .env loading for local runs
    from dotenv import load_dotenv
    import pathlib
    env_path = pathlib.Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except Exception:  # pragma: no cover
    pass

import praw  # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_reddit_sentiment(subreddit: str = "stocks", limit: int = 100) -> Dict[str, Dict[str, Any]]:
    """Fetch ticker sentiment across multiple finance subreddits.

    Returns a mapping: TICKER -> {mentions, compound, positive, neutral, negative, sample_texts, sentiment_score, confidence, sample_size}
    Safe fallbacks ensure an empty dict on credential / network failure.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    # ----- Gather valid ticker universe (fallback list if unavailable) -----
    try:
        from data_feeds.ticker_universe import fetch_ticker_universe
        valid_tickers: Set[str] = set(fetch_ticker_universe(source="finviz", sample_size=200))
    except Exception:
        valid_tickers = {
            "AAPL", "TSLA", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "AMD",
            "META", "NFLX", "SPY", "QQQ", "IWM", "VTI", "SHOP", "ROKU",
            "PLTR", "COIN", "HOOD", "SQ", "PYPL", "DIS", "BABA", "TSM"
        }

    analyzer = SentimentIntensityAnalyzer()
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "oracle-x-bot")
    if not client_id or not client_secret:
        print("[ERROR] Reddit API credentials not set.")
        return {}
    try:
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Failed to initialize Reddit client: {e}")
        return {}

    subreddits = [
        "stocks", "investing", "wallstreetbets", "StockMarket", "options", "pennystocks",
        "SPACs", "RobinHood", "Daytrading", "CryptoCurrency", "personalfinance"
    ]
    extra = os.environ.get("REDDIT_EXTRA_SUBS")
    if extra:
        for s in extra.split(','):
            s = s.strip()
            if s and s not in subreddits:
                subreddits.append(s)

    ticker_pattern = re.compile(r"\b\$?([A-Z]{2,5})\b")
    # Value dicts will later hold both floats and lists (sample_texts); use Any for simplicity
    sentiment_by_ticker: Dict[str, Dict[str, Any]] = {}
    post_texts_by_ticker: Dict[str, List[str]] = {}
    ids_by_ticker: Dict[str, Set[str]] = {}

    DEBUG = os.environ.get("DEBUG_REDDIT", "0") == "1"
    reduced_limit = min(limit, int(os.environ.get("REDDIT_POST_LIMIT", "50")))
    min_upvotes = 0
    listing_mode = os.environ.get("REDDIT_LISTING", "hot")  # hot|new|top
    concurrency = max(1, min(int(os.environ.get("REDDIT_CONCURRENCY", "4")), len(subreddits)))
    early_target = int(os.environ.get("REDDIT_EARLY_MENTION_TARGET", "60"))

    def _fetch_sub(sub: str, listing: str) -> Tuple[str, List[object], Exception | None]:
        try:
            start = time.time()
            if listing == "new":
                it = reddit.subreddit(sub).new(limit=reduced_limit)
            elif listing == "top":
                it = reddit.subreddit(sub).top(limit=reduced_limit, time_filter="day")
            else:
                it = reddit.subreddit(sub).hot(limit=reduced_limit)
            raw = list(it)
            unique = []
            seen = set()
            for p in raw:
                pid = getattr(p, "id", None)
                key = pid if pid is not None else id(p)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(p)
            if DEBUG:
                print(f"[DEBUG][reddit] r/{sub} listing={listing} posts={len(unique)} elapsed={time.time()-start:.2f}s")
            return sub, unique, None
        except Exception as e:  # pragma: no cover
            return sub, [], e

    def _process_posts(posts: List[object]):
        for post in posts:
            try:
                if hasattr(post, 'score') and post.score < min_upvotes:
                    continue
                title = getattr(post, 'title', '') or ''
                body = getattr(post, 'selftext', '') or ''
                text = f"{title} {body}"
                up = text.upper()
                matches = [t for t in set(ticker_pattern.findall(up)) if t in valid_tickers]
                if not matches:
                    continue
                try:
                    sent = analyzer.polarity_scores(text)
                except Exception:
                    continue
                pid_raw = getattr(post, 'id', None) or (hash(title) ^ hash(body))
                pid = str(pid_raw)
                for tk in matches:
                    bucket = ids_by_ticker.setdefault(tk, set())
                    if pid in bucket:
                        continue
                    bucket.add(pid)
                    stats = sentiment_by_ticker.setdefault(tk, {"mentions": 0, "compound": 0.0, "positive": 0.0, "neutral": 0.0, "negative": 0.0})
                    stats["mentions"] += 1
                    stats["compound"] += sent.get("compound", 0.0)
                    stats["positive"] += sent.get("pos", 0.0)
                    stats["neutral"] += sent.get("neu", 0.0)
                    stats["negative"] += sent.get("neg", 0.0)
                    texts = post_texts_by_ticker.setdefault(tk, [])
                    short = text[:200] + "..." if len(text) > 200 else text
                    if len(texts) < 12:
                        texts.append(short.strip())
            except Exception:  # pragma: no cover
                continue

    # ---- First pass (selected listing) ----
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        fut_map = {ex.submit(_fetch_sub, s, listing_mode): s for s in subreddits}
        for fut in as_completed(fut_map):
            _sub, posts, err = fut.result()
            if err and DEBUG:
                print(f"[WARN][reddit] fetch error r/{_sub}: {err}")
            _process_posts(posts)
            if sum(v.get('mentions', 0) for v in sentiment_by_ticker.values()) >= early_target:
                if DEBUG:
                    print(f"[DEBUG][reddit] early exit target={early_target}")
                break

    # ---- Fallback pass (new) if no mentions ----
    if not sentiment_by_ticker and listing_mode != "new":
        if DEBUG:
            print("[DEBUG][reddit] zero mentions first pass; fallback listing=new")
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            fut_map = {ex.submit(_fetch_sub, s, "new"): s for s in subreddits}
            for fut in as_completed(fut_map):
                _sub, posts, err = fut.result()
                if err and DEBUG:
                    print(f"[WARN][reddit] fallback fetch error r/{_sub}: {err}")
                _process_posts(posts)

    # ---- Final aggregation / normalization ----
    for tk, stats in sentiment_by_ticker.items():
        m = int(stats.get("mentions", 0) or 0)
        if m <= 0:
            continue
        stats["compound"] /= m
        stats["positive"] /= m
        stats["neutral"] /= m
        stats["negative"] /= m
        texts = post_texts_by_ticker.get(tk, [])
        stats["sample_texts"] = texts[:5]
        stats["sentiment_score"] = float(stats["compound"])  # align orchestrator expectation
        stats["confidence"] = float(max(0.2, min(0.95, 0.5 + 0.02 * m)))
        stats["sample_size"] = m

    return sentiment_by_ticker


# Legacy helper (unused but kept for backward compatibility / tests referencing it)
def _extracted_from_fetch_reddit_sentiment_45(ticker, sentiment_by_ticker, sentiment):  # pragma: no cover
    if ticker not in sentiment_by_ticker:
        sentiment_by_ticker[ticker] = {"mentions": 0, "compound": 0.0, "positive": 0.0, "neutral": 0.0, "negative": 0.0}
    sentiment_by_ticker[ticker]["mentions"] += 1
    sentiment_by_ticker[ticker]["compound"] += sentiment["compound"]
    sentiment_by_ticker[ticker]["positive"] += sentiment["pos"]
    sentiment_by_ticker[ticker]["neutral"] += sentiment["neu"]
    sentiment_by_ticker[ticker]["negative"] += sentiment["neg"]
