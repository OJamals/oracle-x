"""
Oracle-X GUI - System Monitor Page

Monitor system health, logs, and service status.
"""

import streamlit as st
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def show():
    """Display the system monitor page"""
    
    st.markdown('<h2 class="section-header">🖥️ System Monitor</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Monitor system health, services, logs, and diagnostics.
    """)
    
    # Monitor sections
    tab1, tab2, tab3, tab4 = st.tabs(["🏥 Health Check", "📝 Logs", "🔌 Services", "🛠️ Diagnostics"])
    
    with tab1:
        show_health_check()
    
    with tab2:
        show_logs()
    
    with tab3:
        show_services()
    
    with tab4:
        show_diagnostics()


def show_health_check():
    """Display system health check"""
    
    st.markdown("### 🏥 System Health Check")
    
    if st.button("🔄 Refresh Health Status", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    # Configuration
    st.markdown("#### ⚙️ Configuration")
    
    try:
        from env_config import load_config
        config = load_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Check OpenAI
            if config.get('OPENAI_API_KEY') and config.get('OPENAI_API_KEY') != 'your_openai_api_key_here':
                st.success("✅ OpenAI API Key: Configured")
            else:
                st.error("❌ OpenAI API Key: Not configured")
            
            # Check model
            if config.get('OPENAI_MODEL'):
                st.success(f"✅ Model: {config.get('OPENAI_MODEL')}")
            else:
                st.warning("⚠️ Model: Not specified")
        
        with col2:
            # Check embedding
            if config.get('EMBEDDING_MODEL'):
                st.success(f"✅ Embedding: {config.get('EMBEDDING_MODEL')}")
            else:
                st.warning("⚠️ Embedding: Not specified")
            
            # Check Qdrant
            if config.get('QDRANT_URL'):
                st.info(f"ℹ️ Qdrant URL: {config.get('QDRANT_URL')}")
            else:
                st.warning("⚠️ Qdrant: Not configured")
    
    except Exception as e:
        st.error(f"❌ Configuration Error: {str(e)}")
    
    st.markdown("---")
    
    # Services
    st.markdown("#### 🔌 Services Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Check Qdrant
        try:
            from qdrant_client import QdrantClient
            from env_config import load_config
            config = load_config()
            qdrant_url = config.get('QDRANT_URL', 'http://localhost:6333')
            
            try:
                client = QdrantClient(url=qdrant_url, timeout=2)
                collections = client.get_collections()
                st.success(f"✅ Qdrant: Online ({len(collections.collections)} collections)")
            except Exception as e:
                st.error(f"❌ Qdrant: Offline")
        except ImportError:
            st.warning("⚠️ Qdrant: Client not installed")
    
    with col2:
        # Check Redis (optional)
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=1)
            r.ping()
            st.success("✅ Redis: Online")
        except:
            st.info("ℹ️ Redis: Not running (optional)")
    
    st.markdown("---")
    
    # Directories
    st.markdown("#### 📂 Directory Status")
    
    directories = {
        'playbooks': 'Playbooks',
        'signals': 'Signals',
        'models': 'ML Models',
        'data/databases': 'Databases',
        'logs': 'Logs',
        'backtest_data': 'Backtest Data'
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (dir_path, label) in enumerate(directories.items()):
        target_col = [col1, col2, col3][i % 3]
        
        with target_col:
            if os.path.exists(dir_path):
                # Count files
                if os.path.isdir(dir_path):
                    files = os.listdir(dir_path)
                    st.success(f"✅ {label} ({len(files)} items)")
                else:
                    st.success(f"✅ {label}")
            else:
                st.warning(f"⚠️ {label}: Missing")
    
    st.markdown("---")
    
    # Disk space
    st.markdown("#### 💾 Storage")
    
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        used_pct = (used / total) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Space", f"{total_gb:.1f} GB")
        with col2:
            st.metric("Used Space", f"{used_gb:.1f} GB ({used_pct:.1f}%)")
        with col3:
            st.metric("Free Space", f"{free_gb:.1f} GB")
        
        # Progress bar
        st.progress(used_pct / 100)
    
    except Exception as e:
        st.error(f"❌ Could not get disk usage: {str(e)}")


def show_logs():
    """Display system logs"""
    
    st.markdown("### 📝 System Logs")
    
    # Log viewer
    log_dir = "logs"
    
    if not os.path.exists(log_dir):
        st.warning("⚠️ Logs directory not found")
        return
    
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".log") or f.endswith(".txt")], reverse=True)
    
    if not log_files:
        st.info("No log files found")
        return
    
    # File selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_log = st.selectbox("Select Log File", log_files)
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    if selected_log:
        log_path = os.path.join(log_dir, selected_log)
        
        # File info
        file_size = os.path.getsize(log_path) / 1024  # KB
        mod_time = datetime.fromtimestamp(os.path.getmtime(log_path)).strftime("%Y-%m-%d %H:%M:%S")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.text(f"Size: {file_size:.2f} KB")
        with col_info2:
            st.text(f"Modified: {mod_time}")
        
        # View options
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            tail_lines = st.number_input("Show last N lines", min_value=10, max_value=1000, value=100)
        
        with col_opt2:
            filter_text = st.text_input("Filter text", value="")
        
        with col_opt3:
            search_error = st.checkbox("Show errors only", value=False)
        
        # Read and display log
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            # Apply filters
            if search_error:
                lines = [l for l in lines if 'error' in l.lower() or 'exception' in l.lower() or 'failed' in l.lower()]
            
            if filter_text:
                lines = [l for l in lines if filter_text.lower() in l.lower()]
            
            # Get last N lines
            display_lines = lines[-tail_lines:] if len(lines) > tail_lines else lines
            
            # Display
            st.code(''.join(display_lines), language="text")
            
            # Download button
            st.download_button(
                label="📥 Download Log",
                data=''.join(lines),
                file_name=selected_log,
                mime="text/plain"
            )
        
        except Exception as e:
            st.error(f"❌ Error reading log: {str(e)}")


def show_services():
    """Display services status and control"""
    
    st.markdown("### 🔌 Services Control")
    
    st.markdown("""
    Monitor and control external services used by Oracle-X.
    """)
    
    # Qdrant
    st.markdown("#### 🗄️ Qdrant Vector Database")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        try:
            from qdrant_client import QdrantClient
            from env_config import load_config
            config = load_config()
            qdrant_url = config.get('QDRANT_URL', 'http://localhost:6333')
            
            st.text(f"URL: {qdrant_url}")
            
            try:
                client = QdrantClient(url=qdrant_url, timeout=2)
                collections = client.get_collections()
                st.success(f"✅ Status: Online")
                st.text(f"Collections: {len(collections.collections)}")
                
                # List collections
                if collections.collections:
                    st.markdown("**Available Collections:**")
                    for coll in collections.collections:
                        st.text(f"• {coll.name}")
            
            except Exception as e:
                st.error(f"❌ Status: Offline")
                st.text(f"Error: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Configuration error: {str(e)}")
    
    with col2:
        if st.button("🔄 Check Qdrant", key="check_qdrant"):
            st.rerun()
        
        if st.button("🔧 Restart Qdrant", key="restart_qdrant"):
            st.info("Manual restart: `docker restart qdrant`")
    
    st.markdown("---")
    
    # API Services
    st.markdown("#### 🌐 API Services")
    
    try:
        from env_config import load_config
        config = load_config()
        
        apis = {
            'TwelveData': config.get('TWELVEDATA_API_KEY'),
            'Financial Modeling Prep': config.get('FINANCIALMODELINGPREP_API_KEY'),
            'Finnhub': config.get('FINNHUB_API_KEY'),
            'Alpha Vantage': config.get('ALPHAVANTAGE_API_KEY'),
            'Polygon': config.get('POLYGON_API_KEY')
        }
        
        for api_name, api_key in apis.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if api_key and api_key != f'your_{api_name.lower().replace(" ", "_")}_api_key_here':
                    st.success(f"✅ {api_name}: Configured")
                else:
                    st.warning(f"⚠️ {api_name}: Not configured")
            
            with col2:
                if api_key:
                    st.text(f"{api_key[:8]}...")
    
    except Exception as e:
        st.error(f"❌ Error checking APIs: {str(e)}")


def show_diagnostics():
    """Display system diagnostics"""
    
    st.markdown("### 🛠️ System Diagnostics")
    
    # Python environment
    st.markdown("#### 🐍 Python Environment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text(f"Python Version: {sys.version.split()[0]}")
        st.text(f"Executable: {sys.executable}")
    
    with col2:
        st.text(f"Platform: {sys.platform}")
        import platform
        st.text(f"Architecture: {platform.machine()}")
    
    st.markdown("---")
    
    # Installed packages
    st.markdown("#### 📦 Key Packages")
    
    packages = [
        'streamlit',
        'pandas',
        'openai',
        'qdrant-client',
        'torch',
        'transformers',
        'plotly'
    ]
    
    import importlib
    
    for pkg in packages:
        try:
            module = importlib.import_module(pkg.replace('-', '_'))
            version = getattr(module, '__version__', 'Unknown')
            st.success(f"✅ {pkg}: {version}")
        except ImportError:
            st.warning(f"⚠️ {pkg}: Not installed")
    
    st.markdown("---")
    
    # Validation tools
    st.markdown("#### 🔍 Validation Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Run System Validation", use_container_width=True):
            with st.spinner("Running validation..."):
                try:
                    result = subprocess.run(
                        [sys.executable, "cli_validate.py", "--help"],
                        cwd="/home/runner/work/oracle-x/oracle-x",
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Validation tool available")
                        st.code(result.stdout, language="text")
                    else:
                        st.error("❌ Validation tool error")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("📊 Check Data Feeds", use_container_width=True):
            st.info("Data feed validation coming soon!")
    
    st.markdown("---")
    
    # Quick diagnostics
    st.markdown("#### ⚡ Quick Diagnostics")
    
    if st.button("🚀 Run Full Diagnostic Check", type="primary", use_container_width=True):
        with st.spinner("Running diagnostics..."):
            st.markdown("**Diagnostic Results:**")
            
            # Check directories
            st.markdown("##### 📂 Directories")
            dirs = ['playbooks', 'signals', 'models', 'data', 'logs']
            for d in dirs:
                if os.path.exists(d):
                    st.success(f"✅ {d}/")
                else:
                    st.warning(f"⚠️ {d}/ (missing)")
            
            # Check files
            st.markdown("##### 📄 Critical Files")
            files = ['main.py', 'signals_runner.py', 'oracle_cli.py', 'config_manager.py']
            for f in files:
                if os.path.exists(f):
                    st.success(f"✅ {f}")
                else:
                    st.error(f"❌ {f} (missing)")
            
            # Check configuration
            st.markdown("##### ⚙️ Configuration")
            try:
                from env_config import load_config
                config = load_config()
                st.success("✅ Configuration loaded")
            except:
                st.error("❌ Configuration error")
            
            st.success("🎉 Diagnostic check completed!")
