"""
Oracle-X GUI - Analytics Page

Performance analytics, backtesting results, and system metrics.
"""

import streamlit as st
import os
import json
from datetime import datetime
import pandas as pd

def show():
    """Display the analytics page"""
    
    st.markdown('<h2 class="section-header">📈 Performance Analytics</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    View backtesting results, model performance, and system analytics.
    """)
    
    # Analytics sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Backtest Results", "🤖 Model Performance", "📈 System Metrics"])
    
    with tab1:
        show_overview()
    
    with tab2:
        show_backtest_results()
    
    with tab3:
        show_model_performance()
    
    with tab4:
        show_system_metrics()


def show_overview():
    """Display analytics overview"""
    
    st.markdown("### 📊 Analytics Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Count playbooks
        playbooks_count = len([f for f in os.listdir("playbooks") if f.endswith(".json")]) if os.path.exists("playbooks") else 0
        st.metric("Total Playbooks", playbooks_count)
    
    with col2:
        # Count backtests
        backtest_count = len(os.listdir("backtest_data")) if os.path.exists("backtest_data") else 0
        st.metric("Backtest Files", backtest_count)
    
    with col3:
        # Count models
        models_count = len([d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d)) and d.startswith("checkpoint")]) if os.path.exists("models") else 0
        st.metric("ML Models", models_count)
    
    with col4:
        # Signals count
        signals_count = len([f for f in os.listdir("signals") if f.endswith(".json")]) if os.path.exists("signals") else 0
        st.metric("Signal Files", signals_count)
    
    st.markdown("---")
    
    # Recent activity chart
    st.markdown("### 📅 Recent Activity")
    
    if os.path.exists("playbooks"):
        files = sorted([f for f in os.listdir("playbooks") if f.endswith(".json")], reverse=True)[:30]
        if files:
            dates = []
            for f in files:
                file_path = os.path.join("playbooks", f)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                dates.append(mod_time.date())
            
            # Count by date
            date_counts = {}
            for date in dates:
                date_counts[date] = date_counts.get(date, 0) + 1
            
            df = pd.DataFrame(list(date_counts.items()), columns=['Date', 'Playbooks'])
            df = df.sort_values('Date')
            
            st.line_chart(df.set_index('Date'))
        else:
            st.info("No playbooks available for activity chart")
    else:
        st.warning("Playbooks directory not found")


def show_backtest_results():
    """Display backtest results"""
    
    st.markdown("### 🎯 Backtest Results")
    
    if not os.path.exists("backtest_data"):
        st.warning("⚠️ Backtest data directory not found")
        st.info("Run backtesting from the command line: `python backtest_tracker/backtest.py`")
        return
    
    backtest_files = sorted(os.listdir("backtest_data"), reverse=True)
    
    if not backtest_files:
        st.info("No backtest results found. Run backtesting to generate results.")
        return
    
    # Display backtest files
    st.markdown(f"**Found {len(backtest_files)} backtest files**")
    
    # Show recent backtests
    st.markdown("#### Recent Backtests")
    for f in backtest_files[:10]:
        file_path = os.path.join("backtest_data", f)
        file_size = os.path.getsize(file_path) / 1024  # KB
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.text(f"📄 {f}")
        with col2:
            st.text(f"{file_size:.2f} KB")
        with col3:
            st.text(mod_time)
    
    st.markdown("---")
    
    # Backtest summary
    st.markdown("#### 📊 Backtest Summary")
    st.info("Detailed backtest analytics coming soon. Will include win rate, P&L, Sharpe ratio, etc.")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Run New Backtest"):
            st.info("Running backtest... This feature will execute: `python backtest_tracker/backtest.py`")
    
    with col2:
        if st.button("📊 Generate Report"):
            st.info("Generating report... This feature will execute: `python backtest_tracker/results_analyzer.py`")


def show_model_performance():
    """Display ML model performance"""
    
    st.markdown("### 🤖 ML Model Performance")
    
    # Check for monitoring database
    db_path = "models/monitoring.db"
    
    if os.path.exists(db_path):
        try:
            import sqlite3
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if tables:
                st.success(f"✅ Found monitoring database with {len(tables)} tables")
                
                # Display table selector
                selected_table = st.selectbox("Select Table", tables)
                
                if selected_table:
                    cursor.execute(f"SELECT * FROM {selected_table} LIMIT 100")
                    rows = cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    if rows:
                        df = pd.DataFrame(rows, columns=columns)
                        st.dataframe(df, use_container_width=True)
                        
                        # Show some metrics if available
                        if 'accuracy' in columns or 'score' in columns:
                            st.markdown("#### 📊 Performance Metrics")
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                if 'accuracy' in columns:
                                    avg_accuracy = df['accuracy'].mean() if not df['accuracy'].empty else 0
                                    st.metric("Avg Accuracy", f"{avg_accuracy:.2%}")
                            
                            with metric_col2:
                                st.metric("Total Records", len(df))
                            
                            with metric_col3:
                                if 'timestamp' in columns or 'created_at' in columns:
                                    st.metric("Latest Record", "Recent")
                    else:
                        st.info(f"Table '{selected_table}' is empty")
            else:
                st.warning("No tables found in monitoring database")
            
            conn.close()
        
        except Exception as e:
            st.error(f"❌ Error reading monitoring database: {str(e)}")
    else:
        st.warning("⚠️ Monitoring database not found")
        st.info("Run ML training to generate performance data")
    
    st.markdown("---")
    
    # Model checkpoints
    st.markdown("### 💾 Model Checkpoints")
    
    if os.path.exists("models"):
        checkpoints = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d)) and d.startswith("checkpoint")]
        
        if checkpoints:
            st.markdown(f"**Found {len(checkpoints)} model checkpoints**")
            
            # Show recent checkpoints
            for checkpoint in sorted(checkpoints, reverse=True)[:10]:
                checkpoint_path = os.path.join("models", checkpoint)
                mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path)).strftime("%Y-%m-%d %H:%M:%S")
                
                with st.expander(f"📦 {checkpoint}"):
                    st.text(f"Created: {mod_time}")
                    
                    # List files in checkpoint
                    files = os.listdir(checkpoint_path)
                    st.text(f"Files: {len(files)}")
                    
                    for f in files[:5]:
                        st.text(f"  • {f}")
        else:
            st.info("No model checkpoints found")
    else:
        st.warning("Models directory not found")


def show_system_metrics():
    """Display system performance metrics"""
    
    st.markdown("### 📈 System Metrics")
    
    # Cache performance
    st.markdown("#### 💾 Cache Performance")
    
    cache_db = "data/databases/cache.db"
    
    if os.path.exists(cache_db):
        try:
            import sqlite3
            
            conn = sqlite3.connect(cache_db)
            cursor = conn.cursor()
            
            # Get cache stats
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if tables:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Cache Tables", len(tables))
                
                # Count total cached items
                total_items = 0
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        total_items += count
                    except:
                        pass
                
                with col2:
                    st.metric("Cached Items", total_items)
                
                with col3:
                    cache_size = os.path.getsize(cache_db) / (1024 * 1024)  # MB
                    st.metric("Cache Size", f"{cache_size:.2f} MB")
            
            conn.close()
        
        except Exception as e:
            st.error(f"❌ Error reading cache: {str(e)}")
    else:
        st.warning("Cache database not found")
    
    st.markdown("---")
    
    # API usage
    st.markdown("#### 🔌 API Usage")
    
    st.info("API usage metrics coming soon. Will track API calls, rate limits, and costs.")
    
    # Placeholder metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("API Calls Today", "N/A")
    
    with col2:
        st.metric("Cache Hit Rate", "N/A")
    
    with col3:
        st.metric("Avg Response Time", "N/A")
    
    st.markdown("---")
    
    # Optimization metrics
    st.markdown("#### 🎯 Optimization Metrics")
    
    opt_db = "data/databases/prompt_optimization.db"
    
    if os.path.exists(opt_db):
        st.success("✅ Prompt optimization database found")
        
        try:
            import sqlite3
            
            conn = sqlite3.connect(opt_db)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if tables:
                st.markdown(f"**Optimization tables: {', '.join(tables)}**")
            
            conn.close()
        
        except Exception as e:
            st.error(f"❌ Error reading optimization database: {str(e)}")
    else:
        st.info("Prompt optimization database not found. Enable optimization to generate metrics.")
