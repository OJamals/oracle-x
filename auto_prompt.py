from dashboard.app import auto_generate_market_summary

def get_auto_prompt():
    try:
        return auto_generate_market_summary()
    except Exception:
        return "SPY, QQQ, and major tech stocks in focus. Options flow and sentiment mixed. Awaiting new data."
