# --- Streamlit UI Entrypoint ---
import streamlit as st
import contextlib
import json
import os
import sys
from datetime import datetime
import requests
import plotly.express as px

# Add parent dir to sys.path for backend import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Defer importing the main pipeline runner until the Streamlit UI is actively running
# This prevents importing and triggering Streamlit script-runner context when the
# backend CLI (python main.py) imports the dashboard module.
run_oracle_pipeline = None
try:
    # Only import when running as a Streamlit app (Streamlit sets the "STREAMLIT" env var)
    if os.environ.get("STREAMLIT") or __name__ == "__main__":
        from main import run_oracle_pipeline  # type: ignore
except Exception:
    run_oracle_pipeline = None

# === CONFIG ===
PLAYBOOKS_DIR = "playbooks/"
SIGNALS_DIR = "signals/"
QDRANT_URL = "http://localhost:6333/collections"
QWEN3_URL = "http://localhost:8000/v1/models"

# === UTILS ===
def list_playbooks():
    """Return available playbook files sorted by date."""
    files = [f for f in os.listdir(PLAYBOOKS_DIR) if f.endswith(".json")]
    return sorted(files, reverse=True)

def load_playbook(file_name):
    with open(os.path.join(PLAYBOOKS_DIR, file_name), "r") as f:
        return json.load(f)

def plot_signal_strengths(trades):
    """Plot simple bar chart for scenario probabilities."""
    data = []
    for trade in trades:
        scenarios = trade.get("scenario_tree", {})
        for case, text in scenarios.items():
            with contextlib.suppress(Exception):
                percent = int(text.split("%") [0].strip())
                data.append({
                    "Ticker": trade['ticker'],
                    "Case": case.title(),
                    "Probability": percent
                })
    if data:
        fig = px.bar(
            data,
            x="Ticker",
            y="Probability",
            color="Case",
            barmode="group",
            title="Scenario Probabilities by Trade"
        )
        st.plotly_chart(fig, use_container_width=True)

def get_latest_file_info(directory):
    if not os.path.exists(directory):
        return None, None
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not files:
        return None, None
    latest = max(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    path = os.path.join(directory, latest)
    mod_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    return latest, mod_time

def check_service(url, api_key=None):
    headers = {}
    if api_key:
        headers["api-key"] = api_key
    try:
        resp = requests.get(url, timeout=2, headers=headers)
        print(f"[DEBUG] {url} status_code: {resp.status_code}, response: {resp.text[:200]}")
        if resp.status_code == 200:
            return True
    except Exception as e:
        print(f"[DEBUG] check_service error for {url}: {e}")
    return False

def auto_generate_market_summary():
    # Try to use latest playbook or signals for a summary, else fallback
    try:
        print("[DEBUG] Entered auto_generate_market_summary")
        playbooks = list_playbooks()
        print(f"[DEBUG] Found playbooks: {playbooks}")
        if playbooks:
            return _extracted_from_auto_generate_market_summary_8(playbooks)
    except Exception as e:
        print(f"[DEBUG] Exception in auto_generate_market_summary: {e}")
    return "SPY, QQQ, and major tech stocks in focus. Options flow and sentiment mixed. Awaiting new data."


# TODO Rename this here and in `auto_generate_market_summary`
def _extracted_from_auto_generate_market_summary_8(playbooks):
    latest_playbook = load_playbook(playbooks[0])
    print(f"[DEBUG] Loaded latest playbook: {latest_playbook}")
    tape = latest_playbook.get("tomorrows_tape")
    trades = latest_playbook.get("trades", [])
    tickers = ', '.join([t.get('ticker', '') for t in trades if 'ticker' in t])
    summary = f"Tomorrow's tape: {tape}\nKey trades: {tickers}"
    print(f"[DEBUG] Generated summary: {summary}")
    return summary
if os.environ.get("STREAMLIT") or __name__ == "__main__":
    import streamlit as st

    st.title("🔮 Oracle-X Research & Analysis Dashboard")
    st.markdown("""
    This dashboard lets you run the Oracle-X pipeline, visualize research, and analyze extracted data interactively.
    """)

    if "prompt_text" not in st.session_state:
        st.session_state["prompt_text"] = "TSLA, NVDA, SPY trending bullish after earnings beats. Options sweeps unusually high. Reddit sentiment surging."

    col1, col2 = st.columns([4,1])
    auto_generate = False
    with col2:
        if st.button("Auto-Generate Market Summary", use_container_width=True):
            auto_generate = True

    if auto_generate:
        st.session_state["prompt_text"] = auto_generate_market_summary()

    with col1:
        prompt = st.text_area(
            "Market Summary or Prompt Text",
            value=st.session_state["prompt_text"],
            key="prompt_text_area",
            height=100,
            help="Describe the market, tickers, or sentiment you want to analyze."
        )

    submitted = st.button("Run Oracle-X Pipeline 🚀", use_container_width=True)

    if submitted:
        with st.spinner("Running Oracle-X pipeline. This may take a few moments..."):
            try:
                print("[DEBUG] Submitting to run_oracle_pipeline with prompt:", st.session_state["prompt_text"])
                if not callable(run_oracle_pipeline):
                    st.error("Pipeline runner is not available in this environment. Start the dashboard with `streamlit run dashboard/app.py`.")
                    raise RuntimeError("run_oracle_pipeline not available")
                results = run_oracle_pipeline(st.session_state["prompt_text"])
                print(f"[DEBUG] Pipeline results: {results}")
                st.success(f"Pipeline run for {results['date']}")

                # --- AI-formatted, color-coded, intuitive summary for each trade ---
                st.subheader("Playbook Summary & Visuals")
                playbook = results["playbook"]
                tape = playbook.get("tomorrows_tape")
                if tape:
                    st.markdown(f"<div style='background:#f0f2f6;padding:1em;border-radius:8px;'><b>Tomorrow's Tape:</b> {tape}</div>", unsafe_allow_html=True)

                trades = playbook.get("trades") if isinstance(playbook, dict) else None
                if trades and isinstance(trades, list) and len(trades) > 0:
                    for i, trade in enumerate(trades, 1):
                        ticker = trade.get('ticker', 'N/A')
                        direction = trade.get('direction', '').lower()
                        instrument = trade.get('instrument', '')
                        entry_range = trade.get('entry_range', '')
                        profit_target = trade.get('profit_target', '')
                        stop_loss = trade.get('stop_loss', '')
                        thesis = trade.get('thesis', '')
                        scenario_tree = trade.get('scenario_tree', {})
                        price_chart = trade.get('price_chart')
                        scenario_chart = trade.get('scenario_chart')

                        # Color for direction
                        if direction in ['long', 'buy', 'call', 'bullish']:
                            dir_color = '#2ecc40'  # green
                            dir_label = 'BUY'
                        elif direction in ['short', 'sell', 'put', 'bearish']:
                            dir_color = '#e74c3c'  # red
                            dir_label = 'SELL'
                        else:
                            dir_color = '#888888'
                            dir_label = direction.upper() if direction else 'N/A'

                        st.markdown(f"""
    <div style='border:1px solid #e0e0e0; border-radius:10px; margin:1em 0; padding:1.5em; background:#fff;'>
      <h3 style='margin-bottom:0.5em;'>
        <span style='color:#444;'>{ticker}</span>
        <span style='background:{dir_color};color:#fff;padding:0.2em 0.8em;border-radius:6px;font-size:0.8em;margin-left:1em;'>{dir_label}</span>
        <span style='font-size:0.8em;color:#888;margin-left:1em;'>{instrument}</span>
      </h3>
      <div style='margin-bottom:0.5em;'><b>Entry Range:</b> <span style='color:#444;'>{entry_range}</span> &nbsp;|&nbsp; <b>Profit Target:</b> <span style='color:#2ecc40;'>{profit_target}</span> &nbsp;|&nbsp; <b>Stop Loss:</b> <span style='color:#e74c3c;'>{stop_loss}</span></div>
      <div style='margin-bottom:0.5em;'><b>Thesis:</b> <span style='color:#444;'>{thesis}</span></div>
      <div style='margin-bottom:0.5em;'><b>Scenario Probabilities:</b> """, unsafe_allow_html=True)
                        # Scenario tree as colored chips
                        for case, text in scenario_tree.items():
                            case_color = '#2ecc40' if 'bull' in case else ('#e74c3c' if 'bear' in case else '#8884d8')
                            st.markdown(f"<span style='background:{case_color};color:#fff;padding:0.2em 0.7em;border-radius:5px;margin-right:0.5em;font-size:0.9em;'>{case.replace('_',' ').title()}: {text}</span>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Visuals side by side
                        chart_cols = st.columns(2)
                        if price_chart and os.path.exists(price_chart):
                            chart_cols[0].image(price_chart, caption=f"{ticker} Price Chart", use_container_width=True)
                        if scenario_chart and os.path.exists(scenario_chart):
                            chart_cols[1].image(scenario_chart, caption=f"{ticker} Scenario Chart", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("No trades found in playbook output.")

                # --- Table for trade recommendations (easy to read, color-coded) ---
                if trades and isinstance(trades, list) and len(trades) > 0:
                    import pandas as pd
                    def color_direction(val):
                        if str(val).lower() in {'long', 'buy', 'call', 'bullish'}:
                            return 'background-color: #d4f8e8; color: #218c3a; font-weight: bold;'
                        elif str(val).lower() in {'short', 'sell', 'put', 'bearish'}:
                            return 'background-color: #ffeaea; color: #c0392b; font-weight: bold;'
                        return ''
                    summary_cols = ["ticker", "direction", "instrument", "entry_range", "profit_target", "stop_loss", "thesis"]
                    summary_data = [
                        {col: trade.get(col, "") for col in summary_cols} for trade in trades
                    ]
                    df = pd.DataFrame(summary_data)
                    st.subheader("Trade Recommendations Table")
                    st.dataframe(df.style.map(color_direction, subset=['direction']), use_container_width=True)

                # (Optional) Show raw JSON in expander for advanced users
                with st.expander("Show Raw Playbook JSON"):
                    st.json(results["playbook"])

                # Show logs
                with st.expander("Show Pipeline Logs"):
                    st.code(results["logs"], language="text")

                # Show trades table if available
                if trades and isinstance(trades, list) and len(trades) > 0:
                    st.subheader("Extracted Trades Table")
                    st.dataframe(trades)
            except Exception as e:
                print(f"[DEBUG] Exception in run_oracle_pipeline: {e}")
                st.error(f"An error occurred while running the pipeline: {e}")