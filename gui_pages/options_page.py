"""
Oracle-X GUI - Options Analysis Page

Real-time options analysis and opportunities.
"""

import streamlit as st
import subprocess
import sys
import json
from datetime import datetime

def show():
    """Display the options analysis page"""
    
    st.markdown('<h2 class="section-header">🎯 Options Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Analyze options opportunities with multi-model valuation and ML predictions.
    """)
    
    # Analysis mode
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Single Ticker", "Market Scan", "Position Monitor"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if analysis_mode == "Single Ticker":
        show_single_ticker_analysis()
    elif analysis_mode == "Market Scan":
        show_market_scan()
    elif analysis_mode == "Position Monitor":
        show_position_monitor()


def show_single_ticker_analysis():
    """Single ticker options analysis"""
    
    st.markdown("### 📊 Single Ticker Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL", help="Enter stock ticker to analyze").upper()
    
    with col2:
        min_score = st.number_input("Min Score", min_value=0, max_value=100, value=65, help="Minimum opportunity score")
    
    with col3:
        limit = st.number_input("Max Results", min_value=1, max_value=50, value=10, help="Maximum opportunities to show")
    
    # Risk tolerance
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["conservative", "moderate", "aggressive"],
        value="moderate"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            verbose = st.checkbox("Verbose output", value=False)
            save_output = st.checkbox("Save to file", value=False)
        
        with col_adv2:
            min_days = st.number_input("Min Days to Expiry", value=7)
            max_days = st.number_input("Max Days to Expiry", value=90)
    
    # Analyze button
    if st.button("🔍 Analyze Options", type="primary", use_container_width=True):
        st.markdown("---")
        st.markdown("### 📈 Analysis Results")
        
        with st.spinner(f"Analyzing {ticker} options..."):
            try:
                # Build command
                command = [
                    sys.executable,
                    "oracle_options_cli.py",
                    "analyze",
                    ticker,
                    "--limit", str(limit),
                    "--min-score", str(min_score),
                    "--risk", risk_tolerance
                ]
                
                if verbose:
                    command.append("--verbose")
                
                if save_output:
                    output_file = f"options_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    command.extend(["--output", output_file])
                
                # Execute
                result = subprocess.run(
                    command,
                    cwd="/home/runner/work/oracle-x/oracle-x",
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    st.success(f"✅ Analysis completed for {ticker}!")
                    
                    # Parse output
                    output = result.stdout
                    
                    # Try to extract JSON if saved
                    if save_output:
                        try:
                            with open(output_file, 'r') as f:
                                opportunities = json.load(f)
                            
                            if opportunities:
                                for i, opp in enumerate(opportunities[:5], 1):
                                    with st.expander(f"Opportunity #{i}: {opp.get('symbol')} ${opp.get('contract', {}).get('strike')} {opp.get('contract', {}).get('type', '').upper()}", expanded=(i==1)):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.markdown("#### Trade Details")
                                            st.markdown(f"**Score:** {opp.get('scores', {}).get('opportunity', 0):.1f}/100")
                                            st.markdown(f"**Entry:** ${opp.get('trade', {}).get('entry_price', 0):.2f}")
                                            st.markdown(f"**Target:** ${opp.get('trade', {}).get('target_price', 0):.2f}")
                                            st.markdown(f"**Stop Loss:** ${opp.get('trade', {}).get('stop_loss', 0):.2f}")
                                        
                                        with col2:
                                            st.markdown("#### Risk Metrics")
                                            risk = opp.get('risk', {})
                                            st.markdown(f"**Max Loss:** ${risk.get('max_loss', 0):.2f}")
                                            st.markdown(f"**Expected Return:** {risk.get('expected_return', 0):.1%}")
                                            st.markdown(f"**Win Probability:** {risk.get('probability_of_profit', 0):.1%}")
                                        
                                        # Analysis
                                        analysis = opp.get('analysis', {})
                                        if analysis.get('key_reasons'):
                                            st.markdown("**Key Reasons:**")
                                            for reason in analysis['key_reasons']:
                                                st.markdown(f"• {reason}")
                        
                        except Exception as e:
                            st.warning(f"Could not parse saved output: {e}")
                    
                    # Show raw output
                    with st.expander("📝 Raw Output"):
                        st.code(output, language="text")
                else:
                    st.error(f"❌ Analysis failed")
                    st.code(result.stderr, language="text")
            
            except subprocess.TimeoutExpired:
                st.error("❌ Analysis timed out")
            except FileNotFoundError:
                st.error("❌ oracle_options_cli.py not found")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def show_market_scan():
    """Market-wide options scan"""
    
    st.markdown("### 🌐 Market-Wide Scan")
    
    st.markdown("""
    Scan multiple tickers for options opportunities. This may take several minutes.
    """)
    
    # Symbol selection
    scan_mode = st.radio(
        "Scan Mode",
        ["Top Liquid Options", "Custom Symbol List"],
        horizontal=True
    )
    
    if scan_mode == "Top Liquid Options":
        top_n = st.slider("Number of symbols to scan", min_value=5, max_value=50, value=20)
        symbols = None
    else:
        symbols_input = st.text_area(
            "Symbols (comma-separated)",
            value="AAPL,MSFT,GOOGL,NVDA,TSLA,AMD,META,AMZN",
            help="Enter symbols separated by commas"
        )
        symbols = [s.strip().upper() for s in symbols_input.split(",")]
        top_n = None
    
    # Scan options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.number_input("Min Score", value=75, help="Minimum opportunity score")
    
    with col2:
        risk_tolerance = st.selectbox("Risk Level", ["conservative", "moderate", "aggressive"])
    
    with col3:
        max_results = st.number_input("Max Results", value=20)
    
    # Scan button
    if st.button("🔍 Start Market Scan", type="primary", use_container_width=True):
        st.markdown("---")
        st.markdown("### 📊 Scan Results")
        
        with st.spinner("Scanning market for opportunities..."):
            try:
                command = [
                    sys.executable,
                    "oracle_options_cli.py",
                    "scan",
                    "--min-score", str(min_score),
                    "--risk", risk_tolerance
                ]
                
                if scan_mode == "Top Liquid Options":
                    command.extend(["--top", str(top_n)])
                else:
                    command.extend(["--symbols", ",".join(symbols)])
                
                result = subprocess.run(
                    command,
                    cwd="/home/runner/work/oracle-x/oracle-x",
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0:
                    st.success("✅ Market scan completed!")
                    st.code(result.stdout, language="text")
                else:
                    st.error("❌ Scan failed")
                    st.code(result.stderr, language="text")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def show_position_monitor():
    """Monitor existing positions"""
    
    st.markdown("### 📊 Position Monitor")
    
    st.markdown("""
    Monitor existing options positions and get exit recommendations.
    """)
    
    st.info("📌 Position monitoring coming soon! Upload positions JSON file to track performance.")
    
    # Upload positions file
    uploaded_file = st.file_uploader("Upload Positions JSON", type=["json"])
    
    if uploaded_file:
        try:
            positions = json.load(uploaded_file)
            
            st.success(f"✅ Loaded {len(positions)} positions")
            
            # Display positions
            for i, pos in enumerate(positions, 1):
                with st.expander(f"Position #{i}: {pos.get('symbol')} {pos.get('type', '').upper()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Symbol:** {pos.get('symbol')}")
                        st.markdown(f"**Strike:** ${pos.get('strike')}")
                        st.markdown(f"**Expiry:** {pos.get('expiry')}")
                    
                    with col2:
                        st.markdown(f"**Entry Price:** ${pos.get('entry_price')}")
                        st.markdown(f"**Quantity:** {pos.get('quantity')}")
                        st.markdown(f"**Type:** {pos.get('type', '').upper()}")
            
            # Monitor button
            if st.button("🔍 Monitor Positions", type="primary"):
                st.info("Monitoring positions... Feature coming soon!")
        
        except Exception as e:
            st.error(f"❌ Error loading positions: {str(e)}")
    
    # Manual position entry
    st.markdown("---")
    st.markdown("### ➕ Add Position Manually")
    
    with st.form("add_position"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pos_symbol = st.text_input("Symbol")
            pos_strike = st.number_input("Strike Price", min_value=0.0)
        
        with col2:
            pos_expiry = st.date_input("Expiry Date")
            pos_type = st.selectbox("Type", ["call", "put"])
        
        with col3:
            pos_entry = st.number_input("Entry Price", min_value=0.0)
            pos_qty = st.number_input("Quantity", min_value=1)
        
        submitted = st.form_submit_button("Add Position")
        
        if submitted:
            st.success(f"✅ Position added: {pos_symbol} {pos_type.upper()}")
