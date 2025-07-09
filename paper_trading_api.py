import os
import requests
from datetime import datetime

# Example: Alpaca paper trading API (free tier, requires API key)
# You can swap this for any broker with a REST API

ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

class PaperTradingAPI:
    def __init__(self):
        self.api_key = os.environ.get("ALPACA_API_KEY")
        self.api_secret = os.environ.get("ALPACA_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET must be set in environment.")
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        })

    def submit_order(self, symbol, qty, side, type="market", time_in_force="day"):
        url = f"{ALPACA_BASE_URL}/orders"
        data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force
        }
        try:
            resp = self.session.post(url, json=data, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[ERROR] Paper trade order failed: {e}")
            return None

    def get_account(self):
        url = f"{ALPACA_BASE_URL}/account"
        try:
            resp = self.session.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[ERROR] Paper trading account fetch failed: {e}")
            return None

    def get_positions(self):
        url = f"{ALPACA_BASE_URL}/positions"
        try:
            resp = self.session.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[ERROR] Paper trading positions fetch failed: {e}")
            return None
