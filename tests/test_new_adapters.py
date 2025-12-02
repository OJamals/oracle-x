#!/usr/bin/env python3
"""
Quick test of the newly integrated InvestinyAdapter and StockdexAdapter
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.consolidated_data_feed import ConsolidatedDataFeed
import time


def test_new_adapters():
    feed = ConsolidatedDataFeed()
    print("🧪 Testing newly integrated adapters...")

    # Test Investiny
    print("\n📈 Testing InvestinyAdapter...")
    try:
        quote = feed.investiny.get_quote("AAPL")
        if quote:
            print(f"✅ Investiny quote: AAPL ${quote.price} from {quote.source}")
        else:
            print("❌ Investiny quote failed")
    except Exception as e:
        print(f"❌ Investiny error: {e}")

    time.sleep(2)

    # Test Stockdex
    print("\n📊 Testing StockdexAdapter...")
    try:
        quote = feed.stockdex.get_quote("AAPL")
        if quote:
            print(f"✅ Stockdex quote: AAPL ${quote.price} from {quote.source}")
        else:
            print("❌ Stockdex quote failed")
    except Exception as e:
        print(f"❌ Stockdex error: {e}")

    # Test consolidated fallback
    print("\n🔄 Testing consolidated fallback...")
    try:
        quote = feed.get_quote("AAPL")
        if quote:
            print(f"✅ Consolidated quote: AAPL ${quote.price} from {quote.source}")
        else:
            print("❌ Consolidated quote failed")
    except Exception as e:
        print(f"❌ Consolidated error: {e}")

    # Test financial statements
    print("\n📋 Testing financial statements...")
    try:
        financials = feed.get_financial_statements("AAPL")
        if financials:
            for stmt_type, data in financials.items():
                if data is not None and hasattr(data, "shape"):
                    print(f"✅ {stmt_type}: {data.shape}")
                else:
                    print(f"⚠️ {stmt_type}: No data")
        else:
            print("❌ No financial statements available")
    except Exception as e:
        print(f"❌ Financial statements error: {e}")


if __name__ == "__main__":
    test_new_adapters()
