"""
Oracle-X GUI - Pipeline Runner Page

Interface for running all pipeline modes with configuration options.
"""

import streamlit as st
import subprocess
import sys
import threading
import time
from datetime import datetime

def show():
    """Display the pipeline runner page"""
    
    st.markdown('<h2 class="section-header">▶️ Pipeline Runner</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Execute Oracle-X pipelines with customizable configurations. Monitor progress in real-time.
    """)
    
    # Pipeline selection
    st.markdown("### Select Pipeline Mode")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        pipeline_mode = st.selectbox(
            "Pipeline Type",
            [
                "standard - Main trading playbook generation",
                "enhanced - Enhanced with ML predictions",
                "optimized - With prompt optimization",
                "signals - Market signals collection only",
                "options - Options analysis pipeline",
                "all - Run all pipelines sequentially"
            ],
            index=0,
            key="pipeline_mode_select"
        )
        
        mode = pipeline_mode.split(" - ")[0]
    
    with col2:
        background_mode = st.checkbox("Run in background", value=False, help="Run pipeline in background without blocking UI")
    
    # Pipeline options
    with st.expander("⚙️ Advanced Options", expanded=False):
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            use_cache = st.checkbox("Use cache", value=True, help="Enable caching for faster execution")
            verbose = st.checkbox("Verbose output", value=False, help="Show detailed logs")
        
        with col_opt2:
            save_logs = st.checkbox("Save logs to file", value=True, help="Save execution logs")
            test_mode = st.checkbox("Test mode", value=False, help="Run in test mode with limited data")
    
    # Pipeline descriptions
    st.markdown("---")
    st.markdown("### 📋 Pipeline Details")
    
    descriptions = {
        "standard": {
            "title": "Standard Pipeline",
            "description": "Main Oracle-X pipeline for generating trading playbooks",
            "features": [
                "Market data collection from multiple sources",
                "Sentiment analysis (Reddit, Twitter, News)",
                "Options flow and dark pool analysis",
                "AI-powered scenario generation",
                "Trade recommendation with entry/exit points"
            ],
            "output": "Playbook JSON with trades and market analysis"
        },
        "enhanced": {
            "title": "Enhanced Pipeline",
            "description": "Advanced pipeline with ML predictions and enhanced analytics",
            "features": [
                "All standard pipeline features",
                "Machine learning price predictions",
                "Advanced sentiment aggregation",
                "Technical indicator analysis",
                "Risk assessment and position sizing"
            ],
            "output": "Enhanced playbook with ML predictions"
        },
        "optimized": {
            "title": "Optimized Pipeline",
            "description": "Pipeline with self-learning prompt optimization",
            "features": [
                "All enhanced pipeline features",
                "A/B testing of prompts",
                "Genetic algorithm optimization",
                "Performance tracking and improvement",
                "Automated template evolution"
            ],
            "output": "Optimized playbook with performance metrics"
        },
        "signals": {
            "title": "Signals Collection",
            "description": "Collect and save market signals without generating playbook",
            "features": [
                "Market internals (breadth, advance/decline)",
                "Options flow data",
                "Dark pool activity",
                "Sentiment scores",
                "Earnings calendar"
            ],
            "output": "Daily signals JSON snapshot"
        },
        "options": {
            "title": "Options Analysis",
            "description": "Comprehensive options valuation and opportunity detection",
            "features": [
                "Multi-model valuation (Black-Scholes, Binomial, Monte Carlo)",
                "Greeks calculation (Delta, Gamma, Vega, Theta, Rho)",
                "ML-powered predictions",
                "Risk management and position sizing",
                "Real-time opportunity scanning"
            ],
            "output": "Options opportunities with detailed analysis"
        },
        "all": {
            "title": "Run All Pipelines",
            "description": "Execute all pipelines sequentially",
            "features": [
                "Signals collection",
                "Standard playbook generation",
                "Options analysis",
                "Complete data coverage"
            ],
            "output": "All pipeline outputs"
        }
    }
    
    if mode in descriptions:
        desc = descriptions[mode]
        st.markdown(f"#### {desc['title']}")
        st.markdown(f"*{desc['description']}*")
        
        st.markdown("**Features:**")
        for feature in desc['features']:
            st.markdown(f"• {feature}")
        
        st.markdown(f"**Output:** {desc['output']}")
    
    # Run button
    st.markdown("---")
    
    col_run1, col_run2, col_run3 = st.columns([1, 2, 1])
    
    with col_run2:
        run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
    
    # Execute pipeline
    if run_button:
        st.markdown("---")
        st.markdown("### 📊 Pipeline Execution")
        
        # Build command
        if mode == "signals":
            command = [sys.executable, "signals_runner.py"]
        elif mode == "options":
            command = [sys.executable, "oracle_cli.py", "pipeline", "run", "--mode", "options"]
        elif mode == "all":
            command = [sys.executable, "oracle_cli.py", "pipeline", "run", "--mode", "all"]
        else:
            command = [sys.executable, "main.py", "--mode", mode]
        
        # Add options
        if test_mode:
            command.append("--test")
        
        with st.spinner(f"Running {mode} pipeline..."):
            try:
                # Create placeholder for output
                output_container = st.empty()
                
                start_time = time.time()
                
                if background_mode:
                    # Run in background
                    st.info(f"🔄 Pipeline started in background at {datetime.now().strftime('%H:%M:%S')}")
                    
                    def run_background():
                        subprocess.run(command, cwd="/home/runner/work/oracle-x/oracle-x", capture_output=True)
                    
                    thread = threading.Thread(target=run_background)
                    thread.start()
                    
                    st.success("✅ Pipeline started in background. Check System Monitor for progress.")
                else:
                    # Run in foreground with output
                    result = subprocess.run(
                        command,
                        cwd="/home/runner/work/oracle-x/oracle-x",
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    if result.returncode == 0:
                        st.success(f"✅ Pipeline completed successfully in {elapsed_time:.2f} seconds!")
                        
                        # Show output
                        if verbose or save_logs:
                            with st.expander("📝 Execution Logs", expanded=False):
                                st.code(result.stdout, language="text")
                        
                        # Show errors if any
                        if result.stderr:
                            with st.expander("⚠️ Warnings/Errors", expanded=False):
                                st.code(result.stderr, language="text")
                        
                        # Save logs
                        if save_logs:
                            log_file = f"logs/pipeline_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                            import os
                            os.makedirs("logs", exist_ok=True)
                            with open(log_file, 'w') as f:
                                f.write(f"Pipeline: {mode}\n")
                                f.write(f"Start: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"Duration: {elapsed_time:.2f}s\n")
                                f.write(f"\n--- STDOUT ---\n{result.stdout}\n")
                                f.write(f"\n--- STDERR ---\n{result.stderr}\n")
                            st.info(f"📄 Logs saved to: {log_file}")
                        
                        # Provide next steps
                        st.markdown("### 🎯 Next Steps")
                        next_col1, next_col2 = st.columns(2)
                        
                        with next_col1:
                            if st.button("📊 View Results", key="view_results"):
                                st.session_state.page = 'data'
                                st.rerun()
                        
                        with next_col2:
                            if st.button("📈 View Analytics", key="view_analytics"):
                                st.session_state.page = 'analytics'
                                st.rerun()
                    else:
                        st.error(f"❌ Pipeline failed with exit code {result.returncode}")
                        
                        with st.expander("📝 Error Details", expanded=True):
                            if result.stdout:
                                st.markdown("**Standard Output:**")
                                st.code(result.stdout, language="text")
                            if result.stderr:
                                st.markdown("**Error Output:**")
                                st.code(result.stderr, language="text")
            
            except subprocess.TimeoutExpired:
                st.error("❌ Pipeline timed out after 10 minutes")
            except FileNotFoundError:
                st.error(f"❌ Pipeline script not found. Please check the installation.")
            except Exception as e:
                st.error(f"❌ Error running pipeline: {str(e)}")
    
    # Pipeline history
    st.markdown("---")
    st.markdown("### 📜 Recent Pipeline Runs")
    
    import os
    if os.path.exists("logs"):
        log_files = sorted([f for f in os.listdir("logs") if f.startswith("pipeline_")], reverse=True)[:10]
        if log_files:
            for log_file in log_files:
                file_path = os.path.join("logs", log_file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                
                col_hist1, col_hist2 = st.columns([3, 1])
                with col_hist1:
                    st.text(f"• {log_file}")
                with col_hist2:
                    st.text(mod_time)
        else:
            st.info("No pipeline logs found")
    else:
        st.info("Logs directory not found")
