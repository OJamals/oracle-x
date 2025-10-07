"""
Oracle-X GUI - Data Viewer Page

Browse and visualize playbooks, signals, and historical data.
"""

import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

def show():
    """Display the data viewer page"""
    
    st.markdown('<h2 class="section-header">📊 Data Viewer</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Browse playbooks, signals, and historical trading data.
    """)
    
    # Data type selector
    data_type = st.radio(
        "Select Data Type",
        ["📝 Playbooks", "📡 Signals", "📈 Historical Data", "🗄️ Database Records"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if data_type == "📝 Playbooks":
        show_playbooks()
    elif data_type == "📡 Signals":
        show_signals()
    elif data_type == "📈 Historical Data":
        show_historical_data()
    elif data_type == "🗄️ Database Records":
        show_database_records()


def show_playbooks():
    """Display playbooks viewer"""
    
    st.markdown("### 📝 Trading Playbooks")
    
    if not os.path.exists("playbooks"):
        st.warning("⚠️ Playbooks directory not found")
        return
    
    # List playbooks
    playbook_files = sorted([f for f in os.listdir("playbooks") if f.endswith(".json")], reverse=True)
    
    if not playbook_files:
        st.info("No playbooks found. Run a pipeline to generate playbooks.")
        return
    
    # File selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_file = st.selectbox("Select Playbook", playbook_files)
    
    with col2:
        if st.button("🔄 Refresh List"):
            st.rerun()
    
    if selected_file:
        file_path = os.path.join("playbooks", selected_file)
        
        # File info
        file_stats = os.stat(file_path)
        mod_time = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        file_size = file_stats.st_size / 1024  # KB
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Modified", mod_time)
        with col_info2:
            st.metric("Size", f"{file_size:.2f} KB")
        with col_info3:
            st.metric("File", selected_file)
        
        st.markdown("---")
        
        try:
            with open(file_path, 'r') as f:
                playbook = json.load(f)
            
            # Display playbook content
            tabs = st.tabs(["📊 Overview", "💼 Trades", "🔮 Analysis", "📄 Raw JSON"])
            
            with tabs[0]:
                # Overview
                st.markdown("### Playbook Overview")
                
                # Handle different playbook structures
                if "playbook" in playbook:
                    playbook_content = playbook["playbook"]
                    tape = playbook_content.get("tomorrows_tape")
                    trades = playbook_content.get("trades", [])
                else:
                    tape = playbook.get("tomorrows_tape")
                    trades = playbook.get("trades", [])
                
                if tape:
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>📰 Tomorrow's Tape:</strong><br>
                        {tape}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Summary metrics
                if trades:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Trades", len(trades))
                    with col2:
                        long_count = sum(1 for t in trades if t.get('direction', '').lower() in ['long', 'buy', 'call'])
                        st.metric("Long Positions", long_count)
                    with col3:
                        short_count = sum(1 for t in trades if t.get('direction', '').lower() in ['short', 'sell', 'put'])
                        st.metric("Short Positions", short_count)
            
            with tabs[1]:
                # Trades detail
                st.markdown("### 💼 Trade Details")
                
                if trades and isinstance(trades, list):
                    for i, trade in enumerate(trades, 1):
                        with st.expander(f"Trade #{i}: {trade.get('ticker', 'N/A')} - {trade.get('direction', 'N/A').upper()}", expanded=(i==1)):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**Ticker:** {trade.get('ticker', 'N/A')}")
                                st.markdown(f"**Direction:** {trade.get('direction', 'N/A')}")
                                st.markdown(f"**Instrument:** {trade.get('instrument', 'N/A')}")
                                st.markdown(f"**Entry Range:** {trade.get('entry_range', 'N/A')}")
                            
                            with col2:
                                st.markdown(f"**Profit Target:** {trade.get('profit_target', 'N/A')}")
                                st.markdown(f"**Stop Loss:** {trade.get('stop_loss', 'N/A')}")
                            
                            st.markdown(f"**Thesis:** {trade.get('thesis', 'N/A')}")
                            
                            # Scenario tree
                            scenario_tree = trade.get('scenario_tree', {})
                            if scenario_tree:
                                st.markdown("**Scenario Probabilities:**")
                                for case, prob in scenario_tree.items():
                                    st.markdown(f"- {case.replace('_', ' ').title()}: {prob}")
                            
                            # Charts
                            if trade.get('price_chart') and os.path.exists(trade['price_chart']):
                                st.image(trade['price_chart'], caption=f"{trade.get('ticker')} Price Chart")
                else:
                    st.info("No trades found in playbook")
            
            with tabs[2]:
                # Analysis
                st.markdown("### 🔮 Market Analysis")
                
                # Extract any analysis data
                analysis_data = {}
                for key in playbook.keys():
                    if 'analysis' in key.lower() or 'sentiment' in key.lower():
                        analysis_data[key] = playbook[key]
                
                if analysis_data:
                    for key, value in analysis_data.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}:**")
                        if isinstance(value, dict):
                            st.json(value)
                        else:
                            st.write(value)
                else:
                    st.info("No additional analysis data available")
            
            with tabs[3]:
                # Raw JSON
                st.markdown("### 📄 Raw Playbook Data")
                st.json(playbook)
                
                # Download button
                st.download_button(
                    label="📥 Download Playbook JSON",
                    data=json.dumps(playbook, indent=2),
                    file_name=selected_file,
                    mime="application/json"
                )
        
        except Exception as e:
            st.error(f"❌ Error loading playbook: {str(e)}")


def show_signals():
    """Display signals viewer"""
    
    st.markdown("### 📡 Market Signals")
    
    if not os.path.exists("signals"):
        st.warning("⚠️ Signals directory not found")
        return
    
    # List signals
    signal_files = sorted([f for f in os.listdir("signals") if f.endswith(".json")], reverse=True)
    
    if not signal_files:
        st.info("No signals found. Run signals_runner.py to collect signals.")
        return
    
    # File selector
    selected_file = st.selectbox("Select Signals File", signal_files)
    
    if selected_file:
        file_path = os.path.join("signals", selected_file)
        
        try:
            with open(file_path, 'r') as f:
                signals = json.load(f)
            
            # Display signals
            tabs = st.tabs(["📊 Overview", "📈 Market Internals", "💰 Options Flow", "🌊 Dark Pools", "😊 Sentiment"])
            
            with tabs[0]:
                st.markdown("### Signal Overview")
                
                # Count available signal types
                signal_types = list(signals.keys())
                st.metric("Signal Types", len(signal_types))
                
                st.markdown("**Available Signals:**")
                for signal_type in signal_types:
                    st.markdown(f"- {signal_type.replace('_', ' ').title()}")
            
            with tabs[1]:
                if "market_internals" in signals:
                    st.markdown("### Market Internals")
                    st.json(signals["market_internals"])
                else:
                    st.info("No market internals data")
            
            with tabs[2]:
                if "options_flow" in signals:
                    st.markdown("### Options Flow")
                    st.json(signals["options_flow"])
                else:
                    st.info("No options flow data")
            
            with tabs[3]:
                if "dark_pools" in signals:
                    st.markdown("### Dark Pool Activity")
                    st.json(signals["dark_pools"])
                else:
                    st.info("No dark pool data")
            
            with tabs[4]:
                if "sentiment_data" in signals:
                    st.markdown("### Sentiment Analysis")
                    st.json(signals["sentiment_data"])
                else:
                    st.info("No sentiment data")
            
            # Download button
            st.markdown("---")
            st.download_button(
                label="📥 Download Signals JSON",
                data=json.dumps(signals, indent=2),
                file_name=selected_file,
                mime="application/json"
            )
        
        except Exception as e:
            st.error(f"❌ Error loading signals: {str(e)}")


def show_historical_data():
    """Display historical data viewer"""
    
    st.markdown("### 📈 Historical Data")
    
    st.info("📊 Historical data viewer - Coming soon! Will display backtesting results and performance metrics.")
    
    # Check for backtest data
    if os.path.exists("backtest_data"):
        files = os.listdir("backtest_data")
        if files:
            st.markdown(f"**Found {len(files)} backtest files:**")
            for f in files[:10]:
                st.text(f"• {f}")
    
    if os.path.exists("backtest_tracker"):
        st.markdown("**Backtest tracker available. Run backtesting from Pipeline Runner.**")


def show_database_records():
    """Display database records viewer"""
    
    st.markdown("### 🗄️ Database Records")
    
    # Database selector
    db_options = {
        "Model Monitoring": "models/monitoring.db",
        "Cache": "data/databases/cache.db",
        "Accounts": "data/databases/accounts.db",
        "Prompt Optimization": "data/databases/prompt_optimization.db"
    }
    
    selected_db = st.selectbox("Select Database", list(db_options.keys()))
    db_path = db_options[selected_db]
    
    if os.path.exists(db_path):
        try:
            import sqlite3
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if tables:
                st.markdown(f"**Tables in {selected_db}:**")
                selected_table = st.selectbox("Select Table", tables)
                
                if selected_table:
                    # Get table data
                    cursor.execute(f"SELECT * FROM {selected_table} LIMIT 100")
                    rows = cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    if rows:
                        df = pd.DataFrame(rows, columns=columns)
                        st.dataframe(df, use_container_width=True)
                        
                        # Download CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name=f"{selected_table}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info(f"Table '{selected_table}' is empty")
            else:
                st.warning("No tables found in database")
            
            conn.close()
        
        except Exception as e:
            st.error(f"❌ Error reading database: {str(e)}")
    else:
        st.warning(f"⚠️ Database not found: {db_path}")
