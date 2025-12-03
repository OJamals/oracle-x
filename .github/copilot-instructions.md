## ORACLE-X — AI Agent Development Guide

**Quick Start**: ORACLE-X is an AI-driven trading intelligence engine that transforms multi-source market data (options flow, sentiment, technical indicators) into actionable trading playbooks using LLM-powered analysis and ML predictions.

## 1) Big Picture Architecture

### Core Concept
ORACLE-X operates as a **multi-pipeline trading intelligence system** that:
1. **Ingests**: Real-time market data from 10+ sources (TwelveData, FinViz, Reddit, Twitter/X, GNews, RSS feeds)
2. **Processes**: Through LLM prompt chains (signals → scenario trees → playbook generation)
3. **Enriches**: Via ML ensemble models with genetic optimization
4. **Delivers**: Structured trading playbooks with risk-scored opportunities

### Key Architectural Layers
```
Data Collection (DataFeedOrchestrator)
    ↓ with intelligent caching (469,732x speedup achieved)
LLM Processing (prompt_chain.py)
    ↓ with JSON sanitization and validation
ML Enhancement (ensemble models, genetic optimization)
    ↓ with quality scoring (82.7/100 avg)
Output Generation (Playbooks, Dashboard, CLI)
```

### Critical Directories
- `core/config.py` - **Unified configuration** (replaces old env_config.py, config_manager.py)
- `data_feeds/data_feed_orchestrator.py` - **Central data hub** with caching, fallback, quality validation
- `oracle_engine/prompt_chain.py` - **LLM workflow** (signals → scenarios → playbooks)
- `oracle_options_pipeline.py` - **Options analysis** with Black-Scholes, Monte Carlo valuation
- `data/databases/` - **SQLite DBs** (accounts, model_monitoring, prompt_optimization)
- `tests/integration/` - **Reference patterns** for mocking and testing

## 2) Developer Workflows & Entry Points

### Primary Commands (Most Used)
```bash
# Main trading playbook generation (standard mode)
python main.py

# Unified CLI interface (recommended for automation)
python oracle_cli.py pipeline run --mode standard    # Playbook generation
python oracle_cli.py pipeline run --mode signals     # Data collection only
python oracle_cli.py pipeline run --mode all         # Sequential execution

# Interactive dashboard
streamlit run dashboard/app.py

# Testing (94.4% pass rate - 67/71 tests passing)
pytest tests/                                        # All tests
pytest tests/integration/test_integration_options_pipeline.py  # Options pipeline (18/18 ✅)
```

### Specialized Pipelines
```bash
# Options analysis with valuation models
python oracle_options_cli.py analyze AAPL

# Prompt optimization with genetic algorithms
python oracle_optimize_cli.py analytics

# Data feed validation (reproducible testing)
python scripts/validation/cli_validate.py quote --symbol AAPL
python scripts/validation/cli_validate.py advanced_sentiment --symbol TSLA
```

### Automation Setup (Cron Example)
```cron
# Daily playbook generation at 6PM ET
0 18 * * * cd /path/to/oracle-x && python oracle_cli.py pipeline run --mode all >> cronlog.txt 2>&1
```

## 3) Critical Code Patterns & Gotchas

### Defensive Programming (Essential)
**Best-Effort Enrichment Pattern** - The system never fails on data enrichment:
```python
# ✅ CORRECT - All enrichment functions use defensive try/except
def enrich_playbook_top_level(playbook: dict, orchestrator) -> dict:
    try:
        # Enrichment logic here
        pass
    except Exception as e:
        warnings.warn(f"Enrichment failed: {e}")
        return playbook  # Return original on failure
```
**Never raise exceptions** in:
- `enrich_trades_with_data_feeds()`
- `enrich_playbook_top_level()`
- Any data collection/enrichment flow

### Orchestrator Integration (Critical)
```python
# ✅ CORRECT - Handle orchestrator availability gracefully
try:
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
    orchestrator = DataFeedOrchestrator()
except Exception:
    orchestrator = None  # Many modules check `if orchestrator is None`
```
**Why**: Optional dependencies (TwelveData API, Redis) may be unavailable

### LLM Output Handling (Required)
```python
# ✅ CORRECT - Always sanitize before parsing
def _sanitize_llm_json(text: str) -> str:
    """Remove markdown, comments, trailing commas"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    # ... additional sanitization
    return text

# Then parse
response = llm.generate(prompt)
clean_json = _sanitize_llm_json(response)
data = json.loads(clean_json)
```
**Location**: See `oracle_engine/prompt_chain.py` for reference implementation

### Intelligent Caching System
```python
# Automatic TTL-based caching (5-minute default)
# DataFeedOrchestrator handles this internally:
# - Reddit sentiment: 469,732x speedup achieved
# - Quality metrics: 82.7/100 average
# - Cache invalidation: Automatic on TTL expiry

# ✅ Use CacheService for custom caching
from data_feeds.cache_service import CacheService
cache = CacheService(ttl_seconds=300)
data = cache.get_or_fetch(key, fetch_function)
```

### Configuration Management
```python
# ✅ CORRECT - Use unified config system
from core.config import config

# Access API keys
openai_key = config.model.openai_api_key
twelvedata_key = config.data_sources.twelvedata_api_key

# Database paths
accounts_db = config.database.get_full_path('accounts')
```
**Note**: Old `env_config.py`, `config_manager.py`, and `common_utils.py` have been removed.

## 4) Integration Points & Feature Development

### Adding New Data Sources
Follow the adapter pattern in `data_feeds/`:
```python
# 1. Create adapter implementing source-specific logic
class MyNewAdapter:
    def get_data(self, symbol: str) -> dict:
        # Fetch and normalize data
        return {"price": 100.0, "volume": 1000000}

# 2. Register in DataFeedOrchestrator
# See: data_feeds/data_feed_orchestrator.py:50-80
# Add to __init__ method and fallback chains

# 3. Add quality validation
def validate_quality(self, data: dict) -> float:
    """Return quality score 0-100"""
    completeness = sum(1 for v in data.values() if v) / len(data)
    return completeness * 100
```

### LLM Prompt Chain Development
**3-Stage Pipeline** (see `oracle_engine/prompt_chain.py`):
```python
# Stage 1: Signals → Opportunity Analysis
signals = collect_market_signals()
opportunities = llm.analyze_opportunities(signals)

# Stage 2: Opportunities → Scenario Tree
scenario_tree = llm.generate_scenario_tree(opportunities)

# Stage 3: Scenario Tree → Trading Playbook
playbook = llm.generate_playbook(scenario_tree)
```

**Key Functions**:
- `clean_signals_for_llm()` - Dedupe & truncate signals to prevent prompt bloat
- `extract_scenario_tree()` - Parse LLM JSON with fallback strategies
- Always use `_sanitize_llm_json()` before `json.loads()`

### ML Model Integration
```python
# Model lifecycle (see oracle_engine/ml_model_manager.py)
from oracle_engine.ml_model_manager import MLModelManager

manager = MLModelManager()
model = manager.load_or_train(
    model_type='ensemble',
    features=['momentum', 'sentiment', 'volume'],
    target='price_direction'
)

predictions = model.predict(market_data)
manager.save_checkpoint(model, metrics={'accuracy': 0.82})
```

### Database Integration
```python
# SQLite access patterns
from core.config import config
import sqlite3

# Get database path
db_path = config.database.get_full_path('model_monitoring')

# Use context manager for connections
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE accuracy > ?", (0.7,))
    results = cursor.fetchall()
```

## 5) Environment Variables & Configuration

### Essential API Keys
```bash
# Core LLM (required)
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional custom endpoint

# Market Data (recommended)
export TWELVEDATA_API_KEY="..."  # Primary market data source

# Social Sentiment (optional)
export REDDIT_CLIENT_ID="..."
export REDDIT_CLIENT_SECRET="..."
export TWITTER_BEARER_TOKEN="..."
```

### Configuration Files
```
config/
├── settings.yaml          # Main application settings
├── optimization.env       # Genetic algorithm parameters
├── rss_feeds_config.env   # News feed sources
└── optimization_config.json  # A/B testing thresholds

data/databases/
├── accounts.db           # User preferences
├── model_monitoring.db   # ML performance metrics
└── prompt_optimization.db # Template evolution data
```

### Configuration Access Pattern
```python
from core.config import config

# Hierarchical: env vars → files → defaults
api_key = config.model.openai_api_key
risk_tolerance = config.trading.risk_tolerance
cache_ttl = config.cache.default_ttl_seconds
```

### Key Settings to Know
- **ADVANCED_SENTIMENT_MAX_PER_SOURCE**: Limits sentiment items per source (default: 10)
- **Cache TTL**: 300 seconds (5 min) for market data, 3600 for company info
- **MODEL_NAME**: Affects prompt optimization tracking (set in `agent_bundle/agent.py`)

## 6) Testing & Validation

### Running Tests
```bash
# Full test suite (67/71 passing - 94.4% success rate)
pytest tests/

# Critical integration tests (100% passing)
pytest tests/integration/test_integration_options_pipeline.py  # 18/18 ✅
pytest tests/integration/test_enhanced_pipeline_comprehensive.py
pytest tests/integration/test_fallback_system.py

# Component-specific tests
pytest tests/unit/test_financial_calculator.py
pytest tests/unit/test_enhanced_sentiment_pipeline.py
```

### Test Patterns (Reference: `test_integration_options_pipeline.py`)
```python
# ✅ CORRECT - Use helper for consistent mocking
from tests.integration.test_integration_options_pipeline import initialize_options_model

def test_my_feature():
    orchestrator = Mock(spec=DataFeedOrchestrator)
    model = initialize_options_model(orchestrator)

    # Mock returns expected structure
    result = model.predict('AAPL', mock_contract)
    assert result.price_increase_probability > 0.5
```

### Data Feed Validation (CLI)
```bash
# Test individual adapters with real data
python scripts/validation/cli_validate.py quote --symbol AAPL
python scripts/validation/cli_validate.py advanced_sentiment --symbol TSLA
python scripts/validation/cli_validate.py market_breadth
python scripts/validation/cli_validate.py sector_performance

# Compare values with tolerance
python scripts/validation/cli_validate.py compare --value 195.23 --ref_value 196.5 --tolerance_pct 2.0
```

### Known Issues (Non-Critical)
- 4 test failures: 3 enhanced feature engine edge cases, 1 batch pipeline timeout
- All critical systems (options, data feeds, ML) are 100% functional

## 7) Architecture Deep Dives

### LLM Model Changes
When modifying LLM behavior:
1. Update `MODEL_NAME` in `agent_bundle/agent.py` (affects optimization tracking)
2. Review `oracle_engine/model_attempt_logger.py` (logging compatibility)
3. Update `oracle_engine/prompt_chain.py` sanitization if output format changes

### Vector DB Changes
**Local ChromaDB Storage** (migrated from Qdrant):
1. Update `vector_db/local_store.py` collection configuration
2. Storage location: `data/vector_db/` (automatically created)
3. Uses OpenAI-compatible embeddings via `config.model.embedding_api_base`
4. Test with `vector_db.ensure_collection()` and `vector_db.get_collection_stats()`
5. No external services required - all data stored locally
6. Legacy Qdrant code preserved in `vector_db/qdrant_store.py.backup`

### Data Feed Architecture
**Central Hub Pattern**: All data flows through `DataFeedOrchestrator`:
```
Source Adapters (TwelveData, FinViz, Reddit, etc.)
    ↓
FallbackManager (automatic failover)
    ↓
CacheService (TTL-based caching)
    ↓
QualityValidator (0-100 scoring)
    ↓
DataFeedOrchestrator (unified interface)
```

## 8) Essential Reading (Onboarding Priority)

**Start Here** (15 min):
1. `README.md` - Project overview, setup, commands
2. `.github/copilot-instructions.md` - This file (you are here!)
3. `main.py` lines 1-200 - Pipeline orchestration and entry point

**Architecture Understanding** (30 min):
4. `data_feeds/data_feed_orchestrator.py` lines 1-150 - Central data hub
5. `oracle_engine/prompt_chain.py` lines 1-100 - LLM processing stages
6. `core/config.py` lines 1-100 - Unified configuration system

**Development Patterns** (30 min):
7. `tests/integration/test_integration_options_pipeline.py` - Test patterns & mocking
8. `docs/AGENT_BLUEPRINT.md` - Autonomous agent architecture
9. `spec/Copilot-Processing.md` - Recent system enhancements

## 9) Development Conventions

### Code Style
- **Defensive Programming**: Use try/except with warnings.warn() for enrichment flows
- **Never Fail Silently**: Log failures, but return gracefully degraded results
- **Import Safety**: Always provide fallback stubs for optional dependencies
- **Configuration Access**: Use `core.config` module, never hardcode paths/keys

### Testing Conventions
```python
# ✅ DO - Use helper functions for consistency
from tests.integration.test_integration_options_pipeline import initialize_options_model

# ✅ DO - Mock at the adapter level, not internal methods
orchestrator = Mock(spec=DataFeedOrchestrator)

# ❌ DON'T - Mock internal implementation details
# This makes tests brittle to refactoring
```

### Naming Patterns
- **Pipelines**: `*_pipeline.py` (e.g., `oracle_options_pipeline.py`)
- **Adapters**: `*_adapter.py` (e.g., `twelvedata_adapter.py`)
- **CLIs**: `*_cli.py` (e.g., `oracle_cli.py`)
- **Configs**: `*_config.py` or `*.env` (e.g., `optimization.env`)

### Performance Guidelines
- **Always cache** expensive operations (API calls, ML inference)
- **Use TTL wisely**: 5 min for volatile data, 1 hour for static
- **Batch where possible**: `get_multiple_quotes()` vs repeated `get_quote()`
- **Profile first**: Don't optimize without measuring (use `cProfile`)

## 10) Common Pitfalls & Solutions

### ❌ Problem: Import errors for optional dependencies
```python
# ❌ BAD - Hard failure
from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
```
```python
# ✅ GOOD - Graceful degradation
try:
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator
except ImportError:
    DataFeedOrchestrator = None

# Later: if DataFeedOrchestrator is None: ...
```

### ❌ Problem: LLM returns malformed JSON
```python
# ❌ BAD - Direct parsing
data = json.loads(llm_response)
```
```python
# ✅ GOOD - Sanitize first
from oracle_engine.prompt_chain import _sanitize_llm_json
clean = _sanitize_llm_json(llm_response)
data = json.loads(clean)
```

### ❌ Problem: Missing configuration values
```python
# ❌ BAD - Environment variable access
api_key = os.environ['OPENAI_API_KEY']  # KeyError if missing
```
```python
# ✅ GOOD - Config system with defaults
from core.config import config
api_key = config.model.openai_api_key  # Returns None or default
```

---

**Questions or need clarification?** Check these resources:
- Unclear integration point? See `docs/AGENT_BLUEPRINT.md`
- Test pattern questions? Review `tests/integration/test_integration_options_pipeline.py`
- Configuration questions? Read `core/config.py` docstrings
- Recent changes? See `spec/Copilot-Processing.md`
