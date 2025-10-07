#!/usr/bin/env python3
"""
🔮 Oracle-X Comprehensive GUI Application

A complete graphical interface for the Oracle-X trading intelligence platform.
Provides intuitive access to all functionality, settings, and data visualization.

Features:
- Pipeline execution (all modes)
- Configuration management
- Data viewing (playbooks, signals, historical)
- Performance analytics
- Options analysis
- System monitoring
- Database management

Usage:
    streamlit run gui_app.py
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Page configuration
st.set_page_config(
    page_title="Oracle-X Trading Intelligence Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/OJamals/oracle-x',
        'Report a bug': "https://github.com/OJamals/oracle-x/issues",
        'About': "Oracle-X: AI-Driven Trading Intelligence Platform"
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar navigation
st.sidebar.markdown('<p class="main-header">🔮 Oracle-X</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation menu
pages = {
    "🏠 Home": "home",
    "▶️ Pipeline Runner": "pipeline",
    "⚙️ Settings": "settings",
    "📊 Data Viewer": "data",
    "📈 Analytics": "analytics",
    "🎯 Options Analysis": "options",
    "🖥️ System Monitor": "monitor"
}

selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
st.session_state.page = pages[selected_page]

# Import page modules
from gui_pages import (
    home_page,
    pipeline_page,
    settings_page,
    data_page,
    analytics_page,
    options_page,
    monitor_page
)

# Main content area
st.markdown('<h1 class="main-header">Oracle-X Trading Intelligence Platform</h1>', unsafe_allow_html=True)

# Route to appropriate page
if st.session_state.page == 'home':
    home_page.show()
elif st.session_state.page == 'pipeline':
    pipeline_page.show()
elif st.session_state.page == 'settings':
    settings_page.show()
elif st.session_state.page == 'data':
    data_page.show()
elif st.session_state.page == 'analytics':
    analytics_page.show()
elif st.session_state.page == 'options':
    options_page.show()
elif st.session_state.page == 'monitor':
    monitor_page.show()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p>Oracle-X v2.0</p>
    <p>© 2025 Trading Intelligence</p>
</div>
""", unsafe_allow_html=True)
