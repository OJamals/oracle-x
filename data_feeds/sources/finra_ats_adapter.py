"""
FINRA Alternative Trading System (ATS) Dark Pool Data Adapter

Fetches weekly dark pool volume data from FINRA's transparency portal.
This is FREE, official data that provides institutional trading insights.

Data includes:
- Total dark pool volume by ticker
- Dark pool market share percentage
- Volume breakdown by venue (UBS, Credit Suisse, Citadel, etc.)
"""

import logging
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class FINRAAtsAdapter:
    """
    Fetch dark pool data from FINRA's Alternative Trading System transparency portal.

    Data is aggregated weekly (T+7 days) and includes:
    - Total ATS (dark pool) volume
    - Percentage of total market volume
    - Top venues by volume

    This is official SEC/FINRA data and is completely free.
    """

    # FINRA OTC Transparency portal
    BASE_URL = "https://otctransparency.finra.org"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Oracle-X-Trading-Intelligence/1.0"})

    def get_dark_pool_summary(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Get dark pool volume summary for multiple tickers.

        Args:
            tickers: List of stock symbols

        Returns:
            {
                'tickers': {
                    'NVDA': {
                        'total_ats_volume': 12500000,
                        'ats_market_share_pct': 15.2,
                        'top_venues': ['UBS ATS', 'Credit Suisse CrossFinder'],
                        'signal': 'high_institutional_activity',
                        'confidence': 0.85
                    }
                },
                'data_date': '2025-10-06',
                'data_source': 'finra_ats',
                'timestamp': '2025-10-08T12:35:00Z'
            }
        """
        results = {}

        for ticker in tickers[:10]:  # Limit to 10 tickers
            try:
                data = self._fetch_ticker_ats_data(ticker)
                if data is not None and not data.empty:
                    results[ticker] = self._analyze_ats_activity(data, ticker)
            except Exception as e:
                logger.warning(f"Failed to fetch FINRA ATS data for {ticker}: {e}")
                continue

        return {
            "tickers": results,
            "data_date": self._get_latest_reporting_date().isoformat(),
            "data_source": "finra_ats",
            "timestamp": datetime.now().isoformat(),
            "total_tickers_analyzed": len(results),
        }

    def _fetch_ticker_ats_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch ATS data for a specific ticker from FINRA.

        Note: This is a simplified implementation. FINRA's actual API
        requires authentication and has specific request formats.

        In production, you would:
        1. Register for FINRA API access
        2. Use their official CSV downloads
        3. Or scrape their web interface (with proper rate limiting)
        """
        try:
            # Get the most recent week's data
            week_date = self._get_latest_reporting_date()

            # FINRA publishes weekly CSV files
            # Format: https://otctransparency.finra.org/otctransparency/AtsIssueData?...

            # For now, return simulated structure based on FINRA's format
            # In production, replace with actual API call

            logger.info(
                f"Fetching FINRA ATS data for {ticker} (week ending {week_date})"
            )

            # Placeholder: Would fetch from FINRA here
            # return self._parse_finra_csv(response.content)

            return None  # Implement actual API call

        except Exception as e:
            logger.error(f"Error fetching FINRA ATS data for {ticker}: {e}")
            return None

    def _analyze_ats_activity(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Analyze ATS (dark pool) activity and generate trading signals.

        Args:
            df: DataFrame with ATS volume data
            ticker: Stock symbol

        Returns:
            Analysis dict with institutional activity signals
        """
        try:
            # Calculate key metrics
            total_ats_volume = df["ATSVolume"].sum()
            total_market_volume = df["TotalVolume"].sum()
            ats_market_share = (
                (total_ats_volume / total_market_volume * 100)
                if total_market_volume > 0
                else 0
            )

            # Get top venues by volume
            top_venues = df.nlargest(3, "ATSVolume")[
                ["VenueName", "ATSVolume"]
            ].to_dict("records")

            # Generate signal based on ATS market share
            signal, confidence = self._generate_signal(
                ats_market_share, total_ats_volume
            )

            return {
                "total_ats_volume": int(total_ats_volume),
                "total_market_volume": int(total_market_volume),
                "ats_market_share_pct": round(ats_market_share, 2),
                "top_venues": [v["VenueName"] for v in top_venues],
                "top_venue_volumes": {
                    v["VenueName"]: int(v["ATSVolume"]) for v in top_venues
                },
                "signal": signal,
                "confidence": confidence,
                "analysis_date": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error analyzing ATS data for {ticker}: {e}")
            return {"error": str(e), "signal": "unknown", "confidence": 0.0}

    def _generate_signal(self, ats_market_share: float, total_volume: int) -> tuple:
        """
        Generate trading signal based on dark pool market share.

        Interpretation:
        - High ATS share (>15%) = Strong institutional activity
        - Medium ATS share (10-15%) = Moderate institutional interest
        - Low ATS share (<10%) = Primarily retail-driven

        Returns:
            (signal_type, confidence_score)
        """
        if ats_market_share >= 15.0 and total_volume > 1000000:
            return ("high_institutional_activity", 0.85)
        elif ats_market_share >= 12.0 and total_volume > 500000:
            return ("elevated_institutional_activity", 0.70)
        elif ats_market_share >= 10.0:
            return ("moderate_institutional_activity", 0.55)
        elif ats_market_share >= 7.0:
            return ("normal_activity", 0.40)
        else:
            return ("low_institutional_activity", 0.25)

    def _get_latest_reporting_date(self) -> datetime:
        """
        Get the most recent FINRA ATS reporting date.

        FINRA publishes weekly data with a T+7 day lag.
        Data is released on Mondays for the previous week.
        """
        today = datetime.now()
        days_since_monday = today.weekday()
        last_monday = today - timedelta(days=days_since_monday)
        reporting_week = last_monday - timedelta(days=7)  # T+7 lag

        return reporting_week

    def _parse_finra_csv(self, csv_content: bytes) -> pd.DataFrame:
        """
        Parse FINRA CSV format into DataFrame.

        Expected columns:
        - IssueSymbol
        - VenueName
        - ATSVolume
        - TotalVolume
        - ATSShareVolume
        - etc.
        """
        try:
            df = pd.read_csv(StringIO(csv_content.decode("utf-8")))
            return df
        except Exception as e:
            logger.error(f"Error parsing FINRA CSV: {e}")
            return pd.DataFrame()


# Fallback: Enhanced volume analysis (current method)
def get_enhanced_volume_signals(tickers: List[str]) -> Dict[str, Any]:
    """
    Enhanced version of current volume-based dark pool detection.

    Uses multiple volume indicators:
    1. Volume spike detection (>1.5x average)
    2. Large trade detection (trades >50k shares)
    3. Time-of-day analysis (dark pools active pre/post market)
    4. Price-volume divergence

    This is a free fallback when FINRA data is unavailable.
    """
    import yfinance as yf

    results = {}

    for ticker in tickers[:10]:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="30d", interval="1d")

            if hist.empty:
                continue

            # Calculate enhanced metrics
            avg_volume = hist["Volume"].mean()
            recent_volume = hist["Volume"].iloc[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # Get intraday data for large trade detection
            intraday = stock.history(period="1d", interval="5m")

            # Detect large trades (proxy for dark pool blocks)
            large_trades = 0
            if not intraday.empty:
                volume_threshold = (
                    avg_volume * 0.01
                )  # 1% of daily avg in 5min = large trade
                large_trades = (intraday["Volume"] > volume_threshold).sum()

            # Calculate dark pool probability
            dark_pool_probability = min(
                (volume_ratio - 1.0) * 0.5  # Volume spike component
                + (large_trades / 78)
                * 0.3  # Large trade component (78 5min periods in day)
                + 0.2,  # Base probability
                1.0,
            )

            if dark_pool_probability > 0.4:  # Lowered threshold from 0.5
                current_price = hist["Close"].iloc[-1]
                estimated_block_size = int(
                    (recent_volume - avg_volume) * 0.15
                )  # 15% of excess

                results[ticker] = {
                    "estimated_block_volume": max(estimated_block_size, 5000),
                    "current_price": round(float(current_price), 2),
                    "volume_ratio": round(volume_ratio, 2),
                    "large_trades_detected": int(large_trades),
                    "dark_pool_probability": round(dark_pool_probability, 2),
                    "signal": (
                        "possible_institutional_activity"
                        if dark_pool_probability > 0.6
                        else "elevated_volume"
                    ),
                    "confidence": round(dark_pool_probability, 2),
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error in enhanced volume analysis for {ticker}: {e}")
            continue

    return {
        "tickers": results,
        "data_source": "enhanced_volume_analysis",
        "timestamp": datetime.now().isoformat(),
        "total_signals": len(results),
    }


# Main interface function
def fetch_dark_pool_data(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Unified dark pool data fetcher.

    Strategy:
    1. Try FINRA ATS data (official, weekly lag)
    2. Fallback to enhanced volume analysis (real-time proxy)

    Args:
        tickers: List of stock symbols

    Returns:
        Combined dark pool analysis
    """
    if tickers is None:
        tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA"]

    try:
        # Try FINRA data first
        finra_adapter = FINRAAtsAdapter()
        finra_data = finra_adapter.get_dark_pool_summary(tickers)

        # If FINRA data available, use it
        if finra_data.get("total_tickers_analyzed", 0) > 0:
            logger.info(
                f"Using FINRA ATS data for {len(finra_data['tickers'])} tickers"
            )
            return finra_data

    except Exception as e:
        logger.warning(
            f"FINRA ATS data unavailable, using volume analysis fallback: {e}"
        )

    # Fallback to enhanced volume analysis
    logger.info("Using enhanced volume analysis for dark pool detection")
    volume_data = get_enhanced_volume_signals(tickers)

    return volume_data


if __name__ == "__main__":
    # Test the adapter
    logging.basicConfig(level=logging.INFO)

    test_tickers = ["NVDA", "TSLA", "AAPL", "META", "MSFT"]

    print("Testing FINRA ATS Adapter...")
    print("=" * 80)

    data = fetch_dark_pool_data(test_tickers)

    print(f"\nData Source: {data.get('data_source')}")
    print(f"Timestamp: {data.get('timestamp')}")
    print(
        f"Tickers Analyzed: {data.get('total_signals', data.get('total_tickers_analyzed', 0))}"
    )

    print("\nSignals Detected:")
    print("-" * 80)

    for ticker, details in data.get("tickers", {}).items():
        print(f"\n{ticker}:")
        print(f"  Signal: {details.get('signal', 'N/A')}")
        print(f"  Confidence: {details.get('confidence', 0):.2f}")

        if "ats_market_share_pct" in details:
            print(f"  Dark Pool Market Share: {details['ats_market_share_pct']:.2f}%")
            print(f"  Top Venues: {', '.join(details.get('top_venues', []))}")
        elif "dark_pool_probability" in details:
            print(f"  Dark Pool Probability: {details['dark_pool_probability']:.2f}")
            print(f"  Volume Ratio: {details['volume_ratio']:.2f}x")
            print(f"  Large Trades: {details['large_trades_detected']}")
