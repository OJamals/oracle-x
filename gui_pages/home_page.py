"""
Oracle-X GUI - Home Page

Dashboard overview with system status and quick actions.
"""

import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path

def show():
    """Display the home page"""
    
    st.markdown('<h2 class="section-header">🏠 Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        playbooks_count = len([f for f in os.listdir("playbooks") if f.endswith((".json", ".msgpack"))]) if os.path.exists("playbooks") else 0
        st.metric("📝 Playbooks", playbooks_count)
    
    with col2:
        signals_count = len([f for f in os.listdir("signals") if f.endswith(".json")]) if os.path.exists("signals") else 0
        st.metric("📡 Signals", signals_count)
    
    with col3:
        models_count = len([d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d)) and d.startswith("checkpoint")]) if os.path.exists("models") else 0
        st.metric("🤖 ML Models", models_count)
    
    with col4:
        # Check if config is loaded
        try:
            from config_manager import get_config
            config = get_config()
            config_status = "✅ Loaded"
        except:
            config_status = "⚠️ Not Loaded"
        st.metric("⚙️ Config", config_status)
    
    st.markdown("---")
    
    # System Status
    st.markdown('<h3 class="section-header">System Status</h3>', unsafe_allow_html=True)
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.markdown("#### 🔧 Services")
        
        # Check Qdrant
        try:
            from qdrant_client import QdrantClient
            try:
                from env_config import load_config
                config = load_config()
                qdrant_url = config.get('QDRANT_URL', 'http://localhost:6333')
                client = QdrantClient(url=qdrant_url, timeout=2)
                collections = client.get_collections()
                st.success(f"✅ Qdrant: Online ({len(collections.collections)} collections)")
            except Exception as e:
                st.warning(f"⚠️ Qdrant: Offline or not configured")
        except ImportError:
            st.info("ℹ️ Qdrant: Client not installed")
        
        # Check OpenAI
        try:
            from env_config import load_config
            config = load_config()
            if config.get('OPENAI_API_KEY') and config.get('OPENAI_API_KEY') != 'your_openai_api_key_here':
                st.success("✅ OpenAI: API Key configured")
            else:
                st.warning("⚠️ OpenAI: API Key not configured")
        except:
            st.error("❌ OpenAI: Configuration error")
    
    with status_col2:
        st.markdown("#### 📂 Data Directories")
        
        directories = ['playbooks', 'signals', 'models', 'data/databases']
        for dir_name in directories:
            if os.path.exists(dir_name):
                st.success(f"✅ {dir_name}/")
            else:
                st.warning(f"⚠️ {dir_name}/ (missing)")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown('<h3 class="section-header">Quick Actions</h3>', unsafe_allow_html=True)
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.markdown("#### 🚀 Run Pipeline")
        if st.button("Run Standard Pipeline", key="home_run_standard"):
            st.session_state.page = 'pipeline'
            st.session_state.pipeline_mode = 'standard'
            st.rerun()
        
        if st.button("Run Signals Collection", key="home_run_signals"):
            st.session_state.page = 'pipeline'
            st.session_state.pipeline_mode = 'signals'
            st.rerun()
    
    with action_col2:
        st.markdown("#### 📊 View Data")
        if st.button("Browse Playbooks", key="home_view_playbooks"):
            st.session_state.page = 'data'
            st.session_state.data_view = 'playbooks'
            st.rerun()
        
        if st.button("View Signals", key="home_view_signals"):
            st.session_state.page = 'data'
            st.session_state.data_view = 'signals'
            st.rerun()
    
    with action_col3:
        st.markdown("#### ⚙️ Configuration")
        if st.button("Manage Settings", key="home_settings"):
            st.session_state.page = 'settings'
            st.rerun()
        
        if st.button("System Monitor", key="home_monitor"):
            st.session_state.page = 'monitor'
            st.rerun()
    
    st.markdown("---")
    
    # Recent Activity
    st.markdown('<h3 class="section-header">Recent Activity</h3>', unsafe_allow_html=True)
    
    recent_col1, recent_col2 = st.columns(2)
    
    with recent_col1:
        st.markdown("#### 📝 Latest Playbooks")
        if os.path.exists("playbooks"):
            files = sorted([f for f in os.listdir("playbooks") if f.endswith(".json")], reverse=True)[:5]
            if files:
                for f in files:
                    file_path = os.path.join("playbooks", f)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M")
                    st.text(f"• {f} ({mod_time})")
            else:
                st.info("No playbooks found")
        else:
            st.warning("Playbooks directory not found")
    
    with recent_col2:
        st.markdown("#### 📡 Latest Signals")
        if os.path.exists("signals"):
            files = sorted([f for f in os.listdir("signals") if f.endswith(".json")], reverse=True)[:5]
            if files:
                for f in files:
                    file_path = os.path.join("signals", f)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M")
                    st.text(f"• {f} ({mod_time})")
            else:
                st.info("No signals found")
        else:
            st.warning("Signals directory not found")
    
    # Information box
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ About Oracle-X</strong><br>
        Oracle-X is an AI-driven trading intelligence platform that integrates real-time market data,
        options flow, sentiment analysis, and machine learning to generate actionable trading insights.
        <br><br>
        <strong>Key Features:</strong>
        <ul>
            <li>Multi-mode pipeline execution (Standard, Enhanced, Optimized)</li>
            <li>Real-time data feeds from multiple sources</li>
            <li>Advanced options analysis and valuation</li>
            <li>Machine learning predictions</li>
            <li>Comprehensive backtesting and analytics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
