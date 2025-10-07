# Oracle-X GUI - Quick Start Guide

Get up and running with the Oracle-X Comprehensive GUI in 5 minutes.

## ⚡ Quick Start

### 1. Install Dependencies
```bash
# Navigate to Oracle-X directory
cd /path/to/oracle-x

# Install required packages
pip install -r requirements.txt
```

### 2. Launch the GUI
```bash
# Start the application
streamlit run gui_app.py

# The browser will open automatically at http://localhost:8501
```

### 3. First-Time Setup

#### Configure API Keys (Optional but Recommended)
1. Click **⚙️ Settings** in the sidebar
2. Go to **🔑 API Keys** tab
3. Enter your API keys:
   - **OpenAI API Key** (required for LLM features)
   - **TwelveData** (recommended for market data)
   - **Financial Modeling Prep** (recommended)
   - **Others** as needed

#### Verify System Status
1. Click **🖥️ System Monitor** in the sidebar
2. Review **🏥 Health Check** section
3. Ensure directories are present:
   - ✅ playbooks/
   - ✅ signals/
   - ✅ models/
   - ✅ data/databases/

## 🎯 Common Tasks

### Run Your First Pipeline
1. Click **▶️ Pipeline Runner** in sidebar
2. Select "**signals - Market signals collection only**"
3. Click **🚀 Run Pipeline**
4. Wait for completion (usually 1-2 minutes)
5. Click **📊 View Results** to see collected signals

### View Existing Data
1. Click **📊 Data Viewer** in sidebar
2. Select **📝 Playbooks** or **📡 Signals**
3. Choose a file from the dropdown
4. Explore the data in different tabs
5. Download as JSON if needed

### Monitor System Health
1. Click **🖥️ System Monitor** in sidebar
2. Check **🏥 Health Check** for system status
3. View **📝 Logs** to see recent activity
4. Run **🛠️ Diagnostics** to validate everything

### Analyze Options
1. Click **🎯 Options Analysis** in sidebar
2. Enter a ticker (e.g., "AAPL")
3. Adjust risk tolerance and min score
4. Click **🔍 Analyze Options**
5. Review opportunities with scores and metrics

## 📋 Page Overview

### 🏠 Home Dashboard
**What it shows:**
- Quick metrics (playbooks, signals, models)
- System status (services, directories)
- Recent activity
- Quick action buttons

**Use it to:**
- Get a quick overview of your system
- Jump to common tasks
- Check system health at a glance

### ▶️ Pipeline Runner
**What it does:**
- Executes any Oracle-X pipeline
- Shows real-time progress
- Saves execution logs

**Available Modes:**
- **Standard**: Main trading playbook generation
- **Enhanced**: With ML predictions
- **Optimized**: Self-learning optimization
- **Signals**: Data collection only
- **Options**: Options analysis
- **All**: Run everything

### ⚙️ Settings
**What you can configure:**
- API keys for all services
- System settings (debug, logging)
- Data source preferences
- Advanced optimization parameters

**Tabs:**
- 🔑 API Keys
- 🎛️ System Settings
- 📊 Data Sources
- 🔧 Advanced

### 📊 Data Viewer
**What you can browse:**
- Playbooks (trading recommendations)
- Signals (market data snapshots)
- Historical data (backtest results)
- Database records (SQLite tables)

**Features:**
- Multiple views (overview, details, raw JSON)
- Export as JSON or CSV
- Search and filter

### 📈 Analytics
**What it shows:**
- System-wide performance metrics
- Backtest results and win rates
- ML model performance
- Cache and API usage stats

**Tabs:**
- 📊 Overview
- 🎯 Backtest Results
- 🤖 Model Performance
- 📈 System Metrics

### 🎯 Options Analysis
**What it does:**
- Analyzes options opportunities
- Provides valuation and Greeks
- Uses ML for predictions
- Assesses risk/reward

**Modes:**
- Single Ticker: Deep analysis of one stock
- Market Scan: Find opportunities across symbols
- Position Monitor: Track existing positions

### 🖥️ System Monitor
**What it monitors:**
- Configuration status
- Service connectivity (Qdrant, APIs)
- Directory integrity
- Storage usage
- Logs and diagnostics

**Sections:**
- 🏥 Health Check
- 📝 Logs
- 🔌 Services
- 🛠️ Diagnostics

## 🔄 Typical Workflow

### Daily Trading Workflow
1. **Morning**: Check **🏠 Home** for system status
2. **Pre-Market**: Run **Signals** pipeline to collect data
3. **Market Hours**: Run **Standard** or **Enhanced** pipeline for playbook
4. **View Playbook**: Use **📊 Data Viewer** → Playbooks
5. **Check Analytics**: Review **📈 Analytics** for performance
6. **Evening**: Run **Options Analysis** for next day opportunities

### Analysis Workflow
1. **Collect Data**: Run **Signals** pipeline
2. **Generate Playbook**: Run **Standard** pipeline
3. **View Results**: Check **📊 Data Viewer**
4. **Analyze Options**: Use **🎯 Options Analysis**
5. **Review Performance**: Check **📈 Analytics**
6. **Monitor System**: Verify **🖥️ System Monitor**

### Configuration Workflow
1. **Initial Setup**: Add API keys in **⚙️ Settings**
2. **Customize**: Adjust system settings
3. **Enable Sources**: Select data sources
4. **Verify**: Check **🖥️ System Monitor** → Health Check
5. **Test**: Run **Signals** pipeline
6. **Monitor**: Review logs for any issues

## 💡 Pro Tips

### Performance
- Enable caching in **⚙️ Settings** for faster execution
- Use "Run in background" for long pipelines
- Check **📈 Analytics** → System Metrics for cache performance

### Data Management
- Regularly review **📊 Data Viewer** to clean old files
- Download important playbooks as JSON backups
- Export database tables as CSV for analysis

### Monitoring
- Bookmark **🖥️ System Monitor** for quick health checks
- Filter logs by "error" to find issues quickly
- Run diagnostics after any configuration changes

### Troubleshooting
- If pipeline fails, check **📝 Logs** in System Monitor
- Verify API keys in **⚙️ Settings** → API Keys
- Run **🛠️ Diagnostics** to validate installation
- Check **🏥 Health Check** for service issues

## ⚠️ Common Issues

### "Qdrant: Client not installed"
```bash
pip install qdrant-client
```

### "OpenAI: API Key not configured"
1. Go to **⚙️ Settings** → **🔑 API Keys**
2. Add your OpenAI API key
3. Restart the application

### "Pipeline failed"
1. Check **🖥️ System Monitor** → **📝 Logs**
2. Filter for "error" messages
3. Verify API keys are configured
4. Ensure all directories exist

### "No playbooks found"
1. Run a pipeline first: **▶️ Pipeline Runner** → Run Standard Pipeline
2. Wait for completion
3. Refresh **📊 Data Viewer**

## 🆘 Getting Help

### In the GUI
- Hover over (?) icons for help tooltips
- Check info boxes (blue) for guidance
- Review error messages for troubleshooting hints

### Documentation
- **GUI_README.md**: Complete feature documentation
- **README.md**: Project overview and architecture
- **docs/CLI_CHEATSHEET.md**: Command-line reference
- **docs/CONFIGURATION.md**: Detailed configuration guide

### Resources
- GitHub Issues: Report bugs or request features
- Main README: Architecture and setup instructions
- Copilot Instructions: `.github/copilot-instructions.md`

## 🎓 Learning Path

### Beginner
1. Start with **🏠 Home** to understand the dashboard
2. Run **Signals** pipeline (simplest)
3. View results in **📊 Data Viewer**
4. Check **🖥️ System Monitor** → Health Check

### Intermediate
1. Configure API keys in **⚙️ Settings**
2. Run **Standard** pipeline for playbooks
3. Explore **📈 Analytics** for performance
4. Try **🎯 Options Analysis** on a single ticker

### Advanced
1. Use **Enhanced** or **Optimized** pipelines
2. Perform market-wide options scans
3. Query databases directly
4. Customize settings for your workflow
5. Monitor system metrics for optimization

## 📊 Next Steps

After getting comfortable with the GUI:

1. **Automate**: Schedule pipelines with cron (see main README)
2. **Customize**: Adjust settings for your trading style
3. **Analyze**: Review historical performance in Analytics
4. **Optimize**: Use prompt optimization features
5. **Scale**: Run market scans for multiple symbols

## ✨ Key Features to Explore

- 🎨 **Custom Styling**: Professional UI with color-coded status
- 📥 **Data Export**: Download playbooks, signals, logs as files
- 🔄 **Background Execution**: Run pipelines without blocking
- 📊 **Database Access**: Query SQLite databases directly
- 🎯 **Options Valuation**: Multi-model pricing with Greeks
- 📈 **Performance Tracking**: Monitor cache hits and API usage
- 🛠️ **Diagnostics**: Built-in validation and troubleshooting

---

**Ready to start?** Run `streamlit run gui_app.py` and begin your Oracle-X journey! 🚀
