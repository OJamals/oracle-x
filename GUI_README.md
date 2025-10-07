# Oracle-X Comprehensive GUI

A complete graphical interface for the Oracle-X trading intelligence platform, providing intuitive access to all system functionality through a modern web-based UI.

## 🚀 Features

### 🏠 Home Dashboard
- System status overview with real-time metrics
- Quick access to all major functions
- Recent activity tracking (playbooks, signals)
- Service health indicators
- Directory status checks

### ▶️ Pipeline Runner
- Execute all pipeline modes:
  - **Standard**: Main trading playbook generation
  - **Enhanced**: With ML predictions and advanced analytics
  - **Optimized**: Self-learning prompt optimization
  - **Signals**: Market data collection only
  - **Options**: Options analysis pipeline
  - **All**: Run all pipelines sequentially
- Real-time execution monitoring
- Configurable options (cache, verbose, test mode)
- Background execution support
- Automatic log saving
- Detailed execution history

### ⚙️ Settings & Configuration
- **API Keys Management**: Configure all data source API keys
  - OpenAI/LLM settings
  - Market data APIs (TwelveData, FMP, Finnhub, etc.)
  - Social media APIs (Reddit, Twitter)
  - Vector database (Qdrant)
- **System Settings**: Debug mode, log levels, performance tuning
- **Data Sources**: Enable/disable individual data sources
- **Advanced Options**: Optimization parameters, cache warming
- Configuration file viewer with download capability

### 📊 Data Viewer
- **Playbooks**: Browse and visualize trading playbooks
  - Overview with Tomorrow's Tape
  - Detailed trade analysis
  - Scenario probabilities
  - Charts and visualizations
  - JSON export
- **Signals**: View market signals data
  - Market internals
  - Options flow
  - Dark pool activity
  - Sentiment analysis
- **Historical Data**: Access backtest results
- **Database Records**: Query SQLite databases
  - Model monitoring
  - Cache data
  - Optimization records

### 📈 Performance Analytics
- **Overview**: System-wide performance metrics
- **Backtest Results**: Historical playbook performance
- **Model Performance**: ML model accuracy and metrics
- **System Metrics**: Cache performance, API usage, optimization stats

### 🎯 Options Analysis
- **Single Ticker Analysis**: Detailed options valuation
  - Multi-model pricing (Black-Scholes, Binomial, Monte Carlo)
  - Greeks calculation
  - ML predictions
  - Risk assessment
- **Market Scan**: Find opportunities across multiple tickers
- **Position Monitor**: Track existing options positions

### 🖥️ System Monitor
- **Health Check**: Real-time system health status
  - Configuration validation
  - Service connectivity
  - Directory integrity
  - Storage monitoring
- **Logs Viewer**: Browse and filter system logs
  - Tail log files
  - Filter by text or error
  - Download logs
- **Services Control**: Monitor external services
  - Qdrant vector database
  - API services status
- **Diagnostics**: System validation and troubleshooting
  - Python environment info
  - Package versions
  - Quick diagnostic tools

## 📦 Installation

### Prerequisites
- Python 3.8+
- All Oracle-X dependencies installed

### Install Additional Dependencies
```bash
# Update requirements
pip install -r requirements.txt

# This includes:
# - streamlit>=1.28.0
# - matplotlib>=3.7.0
# - yfinance>=0.2.0
# - requests>=2.31.0
```

## 🎯 Usage

### Launch the GUI
```bash
# Start the comprehensive GUI
streamlit run gui_app.py

# The application will open in your default browser
# Default URL: http://localhost:8501
```

### Navigation
Use the sidebar menu to navigate between pages:
- 🏠 **Home**: Dashboard overview
- ▶️ **Pipeline Runner**: Execute pipelines
- ⚙️ **Settings**: Configuration management
- 📊 **Data Viewer**: Browse data
- 📈 **Analytics**: Performance metrics
- 🎯 **Options Analysis**: Options trading
- 🖥️ **System Monitor**: Health & diagnostics

### First-Time Setup
1. **Configure API Keys**: Go to Settings → API Keys
2. **Verify Services**: Check System Monitor → Health Check
3. **Run Test Pipeline**: Pipeline Runner → Run Signals Collection
4. **View Results**: Data Viewer → Signals

## 🔧 Configuration

The GUI reads configuration from:
- `.env` - Main environment variables
- `optimization.env` - Optimization settings
- `rss_feeds_config.env` - RSS feed configuration
- `config_manager.py` - Centralized configuration loader

## 📊 Page Details

### Home Dashboard
- Quick metrics: Playbooks, Signals, Models, Config status
- Service status: Qdrant, OpenAI connectivity
- Recent activity: Latest files and timestamps
- Quick actions: Jump to any major function

### Pipeline Runner
Execute any pipeline with detailed configuration:
- Mode selection with descriptions
- Advanced options (cache, verbose, test mode)
- Real-time progress monitoring
- Execution logs with filtering
- Background execution support
- Pipeline history tracking

### Settings Page
Four main sections:
1. **API Keys**: All API credentials in one place
2. **System Settings**: Performance and logging options
3. **Data Sources**: Enable/disable individual sources
4. **Advanced**: Optimization and cache configuration

### Data Viewer
Four data types:
1. **Playbooks**: Complete playbook visualization
2. **Signals**: Market signals breakdown
3. **Historical**: Backtest results (coming soon)
4. **Database Records**: Direct database access

### Analytics Page
Four analytics sections:
1. **Overview**: System-wide metrics and activity
2. **Backtest Results**: Performance analysis
3. **Model Performance**: ML metrics and checkpoints
4. **System Metrics**: Cache, API, optimization stats

### Options Analysis
Three analysis modes:
1. **Single Ticker**: Detailed options analysis for one symbol
2. **Market Scan**: Find opportunities across multiple tickers
3. **Position Monitor**: Track existing positions

### System Monitor
Four monitoring sections:
1. **Health Check**: Real-time system status
2. **Logs**: Browse and filter system logs
3. **Services**: External service monitoring
4. **Diagnostics**: Validation and troubleshooting

## 🎨 Features

### User Experience
- **Responsive Layout**: Adapts to different screen sizes
- **Custom Styling**: Professional color scheme and typography
- **Real-time Updates**: Live status monitoring
- **Error Handling**: Graceful error messages with troubleshooting hints
- **Download Support**: Export data as JSON, CSV, or logs

### Performance
- **Lazy Loading**: Pages load on demand
- **Caching**: Streamlit caching for better performance
- **Async Operations**: Background pipeline execution
- **Progress Indicators**: Real-time progress bars and spinners

### Security
- **Password Fields**: API keys shown as password inputs
- **No Credential Storage**: All credentials in environment files
- **Read-Only Views**: Safe data browsing without modification

## 🆚 Comparison with Existing Dashboard

### Original Dashboard (`dashboard/app.py`)
- Basic playbook viewer
- Single-page application
- Manual pipeline execution
- Limited functionality

### New Comprehensive GUI (`gui_app.py`)
- Multi-page application with navigation
- Complete system management
- All pipeline modes
- Configuration management
- Data browsing and export
- Performance analytics
- System monitoring
- Options analysis

## 🔄 Migration

The new GUI is designed to coexist with the existing dashboard:

```bash
# Original dashboard (still available)
streamlit run dashboard/app.py

# New comprehensive GUI (recommended)
streamlit run gui_app.py
```

## 🐛 Troubleshooting

### GUI Won't Start
```bash
# Check if streamlit is installed
pip install streamlit

# Verify Python version
python --version  # Should be 3.8+

# Check for errors
streamlit run gui_app.py --logger.level=debug
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify all packages
python -c "import streamlit, pandas, plotly; print('OK')"
```

### Configuration Issues
1. Check `.env` file exists (copy from `.env.example`)
2. Verify API keys are set
3. Run System Monitor → Diagnostics for full check

### Services Offline
- **Qdrant**: Start with `docker start qdrant` or check connection
- **Redis**: Optional service, not required
- **APIs**: Verify API keys in Settings

## 📚 Additional Resources

- **Main README**: `/README.md` - Project overview
- **CLI Cheatsheet**: `/docs/CLI_CHEATSHEET.md` - Command-line reference
- **Configuration Guide**: `/docs/CONFIGURATION.md` - Detailed config docs
- **Copilot Instructions**: `/.github/copilot-instructions.md` - System architecture

## 🤝 Contributing

Improvements welcome! The GUI is modular:
- `gui_app.py` - Main application and navigation
- `gui_pages/*.py` - Individual page modules

To add a new page:
1. Create `gui_pages/new_page.py` with `show()` function
2. Add import to `gui_pages/__init__.py`
3. Add navigation entry in `gui_app.py`

## 📝 License

Same as Oracle-X main project. See main README for details.

## ⚠️ Disclaimer

This GUI is for research and educational purposes. Not financial advice.
Always conduct your own due diligence before making investment decisions.
