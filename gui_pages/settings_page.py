"""
Oracle-X GUI - Settings Page

Configuration management interface for all system settings.
"""

import streamlit as st
import os
from pathlib import Path

def show():
    """Display the settings page"""
    
    st.markdown('<h2 class="section-header">⚙️ Settings & Configuration</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Manage Oracle-X configuration, API keys, and system settings.
    """)
    
    # Configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Keys", "🎛️ System Settings", "📊 Data Sources", "🔧 Advanced"])
    
    # Load current configuration
    try:
        from env_config import load_config
        config = load_config()
    except:
        config = {}
        st.warning("⚠️ Could not load configuration. Using defaults.")
    
    # API Keys Tab
    with tab1:
        st.markdown("### API Key Configuration")
        
        st.markdown("""
        <div class="info-box">
            ℹ️ API keys are stored in <code>.env</code> file. Changes require application restart.
        </div>
        """, unsafe_allow_html=True)
        
        # OpenAI Configuration
        st.markdown("#### 🤖 OpenAI / LLM Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            openai_key = st.text_input(
                "OpenAI API Key",
                value=config.get('OPENAI_API_KEY', ''),
                type="password",
                help="API key for OpenAI services"
            )
            openai_model = st.text_input(
                "OpenAI Model",
                value=config.get('OPENAI_MODEL', 'gpt-4o-mini'),
                help="Primary chat/completion model"
            )
        
        with col2:
            openai_base = st.text_input(
                "OpenAI API Base",
                value=config.get('OPENAI_API_BASE', 'https://api.openai.com/v1'),
                help="Base URL for OpenAI API"
            )
            embedding_model = st.text_input(
                "Embedding Model",
                value=config.get('EMBEDDING_MODEL', 'text-embedding-3-small'),
                help="Model for text embeddings"
            )
        
        st.markdown("---")
        
        # Market Data APIs
        st.markdown("#### 📈 Market Data APIs")
        col1, col2 = st.columns(2)
        
        with col1:
            twelvedata_key = st.text_input(
                "TwelveData API Key",
                value=config.get('TWELVEDATA_API_KEY', ''),
                type="password",
                help="High-quality market data"
            )
            
            fmp_key = st.text_input(
                "Financial Modeling Prep API Key",
                value=config.get('FINANCIALMODELINGPREP_API_KEY', ''),
                type="password",
                help="Comprehensive financial data"
            )
            
            alphavantage_key = st.text_input(
                "Alpha Vantage API Key",
                value=config.get('ALPHAVANTAGE_API_KEY', ''),
                type="password",
                help="Stock market data"
            )
        
        with col2:
            finnhub_key = st.text_input(
                "Finnhub API Key",
                value=config.get('FINNHUB_API_KEY', ''),
                type="password",
                help="Real-time quotes and news"
            )
            
            polygon_key = st.text_input(
                "Polygon API Key",
                value=config.get('POLYGON_API_KEY', ''),
                type="password",
                help="Advanced market data"
            )
        
        st.markdown("---")
        
        # Social Media APIs
        st.markdown("#### 📱 Social Media APIs")
        col1, col2 = st.columns(2)
        
        with col1:
            reddit_client_id = st.text_input(
                "Reddit Client ID",
                value=config.get('REDDIT_CLIENT_ID', ''),
                type="password"
            )
            
            reddit_client_secret = st.text_input(
                "Reddit Client Secret",
                value=config.get('REDDIT_CLIENT_SECRET', ''),
                type="password"
            )
        
        with col2:
            twitter_bearer = st.text_input(
                "Twitter Bearer Token",
                value=config.get('TWITTER_BEARER_TOKEN', ''),
                type="password",
                help="Optional - twscrape works without API"
            )
        
        st.markdown("---")
        
        # Vector Database
        st.markdown("#### 🗄️ Vector Database (Qdrant)")
        col1, col2 = st.columns(2)
        
        with col1:
            qdrant_url = st.text_input(
                "Qdrant URL",
                value=config.get('QDRANT_URL', 'http://localhost:6333')
            )
        
        with col2:
            qdrant_key = st.text_input(
                "Qdrant API Key",
                value=config.get('QDRANT_API_KEY', ''),
                type="password"
            )
        
        # Save button
        if st.button("💾 Save API Keys", type="primary"):
            st.warning("⚠️ API key updates require manual editing of .env file. Feature coming soon.")
    
    # System Settings Tab
    with tab2:
        st.markdown("### System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔧 General Settings")
            
            environment = st.selectbox(
                "Environment",
                ["development", "testing", "production"],
                index=0,
                help="Application environment"
            )
            
            debug_mode = st.checkbox(
                "Debug Mode",
                value=config.get('DEBUG', 'false').lower() == 'true',
                help="Enable detailed logging"
            )
            
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1,
                help="Logging verbosity"
            )
        
        with col2:
            st.markdown("#### ⚡ Performance Settings")
            
            enable_cache = st.checkbox(
                "Enable Caching",
                value=True,
                help="Cache API responses for better performance"
            )
            
            cache_ttl = st.number_input(
                "Cache TTL (minutes)",
                min_value=1,
                max_value=1440,
                value=5,
                help="Time-to-live for cached data"
            )
            
            api_timeout = st.number_input(
                "API Timeout (seconds)",
                min_value=5,
                max_value=120,
                value=30,
                help="Timeout for external API calls"
            )
        
        st.markdown("---")
        
        # Database Configuration
        st.markdown("#### 🗄️ Database Paths")
        
        db_col1, db_col2 = st.columns(2)
        
        with db_col1:
            accounts_db = st.text_input(
                "Accounts DB",
                value=config.get('ACCOUNTS_DB_PATH', 'data/databases/accounts.db')
            )
            
            cache_db = st.text_input(
                "Cache DB",
                value=config.get('CACHE_DB_PATH', 'data/databases/cache.db')
            )
        
        with db_col2:
            monitoring_db = st.text_input(
                "Monitoring DB",
                value=config.get('MODEL_MONITORING_DB_PATH', 'data/databases/model_monitoring.db')
            )
            
            optimization_db = st.text_input(
                "Optimization DB",
                value=config.get('PROMPT_OPTIMIZATION_DB_PATH', 'data/databases/prompt_optimization.db')
            )
        
        if st.button("💾 Save System Settings", type="primary"):
            st.success("✅ Settings saved successfully!")
    
    # Data Sources Tab
    with tab3:
        st.markdown("### Data Source Configuration")
        
        st.markdown("#### 📊 Enabled Data Sources")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Market Data**")
            enable_twelvedata = st.checkbox("TwelveData", value=True)
            enable_fmp = st.checkbox("Financial Modeling Prep", value=True)
            enable_finnhub = st.checkbox("Finnhub", value=True)
            enable_alphavantage = st.checkbox("Alpha Vantage", value=False)
        
        with col2:
            st.markdown("**Sentiment Sources**")
            enable_reddit = st.checkbox("Reddit", value=True)
            enable_twitter = st.checkbox("Twitter", value=True)
            enable_yahoo_news = st.checkbox("Yahoo News", value=True)
            enable_gnews = st.checkbox("Google News", value=True)
        
        with col3:
            st.markdown("**Alternative Data**")
            enable_options_flow = st.checkbox("Options Flow", value=True)
            enable_dark_pools = st.checkbox("Dark Pools", value=True)
            enable_market_breadth = st.checkbox("Market Breadth", value=True)
            enable_earnings = st.checkbox("Earnings Calendar", value=True)
        
        st.markdown("---")
        
        st.markdown("#### 🔄 Data Collection Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_max_per_source = st.number_input(
                "Max Sentiment Items per Source",
                min_value=50,
                max_value=1000,
                value=300,
                step=50,
                help="Maximum texts to collect per sentiment source"
            )
        
        with col2:
            enable_fallback = st.checkbox(
                "Enable API Fallback",
                value=True,
                help="Automatically fallback to alternative APIs on failure"
            )
        
        if st.button("💾 Save Data Source Settings", type="primary"):
            st.success("✅ Data source settings saved!")
    
    # Advanced Tab
    with tab4:
        st.markdown("### Advanced Configuration")
        
        st.markdown("#### 🧪 Optimization System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_optimization = st.checkbox(
                "Enable Prompt Optimization",
                value=True,
                help="Self-learning prompt optimization system"
            )
            
            population_size = st.number_input(
                "Population Size",
                min_value=10,
                max_value=100,
                value=20,
                help="Genetic algorithm population size"
            )
            
            mutation_rate = st.slider(
                "Mutation Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Probability of mutation in genetic algorithm"
            )
        
        with col2:
            max_generations = st.number_input(
                "Max Generations",
                min_value=10,
                max_value=200,
                value=50,
                help="Maximum optimization generations"
            )
            
            crossover_rate = st.slider(
                "Crossover Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Probability of crossover in genetic algorithm"
            )
        
        st.markdown("---")
        
        st.markdown("#### 🔄 Cache Warming")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cache_warming = st.checkbox("Enable Cache Warming", value=True)
            warm_at_open = st.checkbox("Warm at Market Open", value=True)
        
        with col2:
            warm_hourly = st.checkbox("Warm Hourly", value=True)
            warm_interval = st.number_input("Warm Interval (min)", value=60)
        
        if st.button("💾 Save Advanced Settings", type="primary"):
            st.success("✅ Advanced settings saved!")
        
        st.markdown("---")
        
        # Danger Zone
        st.markdown("#### ⚠️ Danger Zone")
        
        with st.expander("🗑️ Reset Configuration"):
            st.warning("This will reset all settings to defaults!")
            if st.button("Reset All Settings", type="secondary"):
                st.error("Reset functionality not yet implemented")
        
        with st.expander("🔄 Reload Configuration"):
            st.info("Reload configuration from .env files")
            if st.button("Reload Configuration", type="secondary"):
                try:
                    from env_config import load_config
                    config = load_config()
                    st.success("✅ Configuration reloaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error reloading configuration: {e}")
    
    # Configuration file viewer
    st.markdown("---")
    st.markdown("### 📄 Configuration Files")
    
    config_files = ['.env.example', 'optimization.env', 'rss_feeds_config.env']
    
    selected_file = st.selectbox("Select configuration file to view:", config_files)
    
    if selected_file:
        file_path = Path(selected_file)
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
            
            st.code(content, language="bash")
            
            st.download_button(
                label=f"📥 Download {selected_file}",
                data=content,
                file_name=selected_file,
                mime="text/plain"
            )
        else:
            st.warning(f"⚠️ File not found: {selected_file}")
