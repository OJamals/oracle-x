# ORACLE-X

## Project Overview
ORACLE-X is an AI-driven trading scenario engine that integrates real-time market data, options flow, sentiment, and anomaly detection to generate actionable playbooks and dashboards for traders.

## Structure
- **main.py**: Main entry point for generating daily playbooks using the agent pipeline.
- **signals_runner.py**: Collects and saves daily signals from all data feeds.
- **dashboard/app.py**: Streamlit dashboard for visualizing playbooks and signals.
- **backtest_tracker/**: Tools for backtesting and analyzing playbook performance.
- **data_feeds/**: Modular scrapers and stubs for market data, options flow, sentiment, etc.
- **oracle_engine/**: Core agent logic, prompt chains, and scenario tree generation.
- **vector_db/**: Qdrant vector store integration for scenario recall and prompt boosting.
- **config/**: Configuration files (e.g., settings.yaml).
- **docs/**: Project documentation and SOPs.

## Setup
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Configure API keys and endpoints in `config/settings.yaml` as needed.
3. Run the signals scraper:
   ```sh
   python signals_runner.py
   ```
4. Generate a playbook:
   ```sh
   python main.py
   ```
5. Launch the dashboard:
   ```sh
   streamlit run dashboard/app.py
   ```

## Notes
- Replace all TODOs in data_feeds with real data sources for production use.
- Backtesting tools are available in `backtest_tracker/` but not integrated into the main pipeline by default.
- See `docs/` for detailed SOPs and architecture.

## ⚙️ How to Run

### 1️⃣ Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

2️⃣ Set Your OpenAI API Key
bash
Copy
Edit
export OPENAI_API_KEY="YOUR_API_KEY_HERE"  # Mac/Linux
# OR
setx OPENAI_API_KEY "YOUR_API_KEY_HERE"    # Windows

3️⃣ Place a Chart Image
Add your overnight chart screenshot (e.g., futures, dark pool flow) to your project folder.
Update example_chart.png in main.py with your actual file name.

4️⃣ Run It
bash
Copy
Edit
python main.py
You’ll see your daily ORACLE-X AGENT PLAYBOOK in your console, with:

1–3 best trades

Entries, profit targets, stop-losses

Scenario tree

Tomorrow’s Tape summary


✅ 5️⃣ 🕒 Hook your pipeline into a cron job (Mac/Linux)
1️⃣ Edit your cron:

bash
Copy
Edit
crontab -e
2️⃣ Add:

cron
Copy
Edit
# Run Oracle-X daily at 6PM Eastern (adjust as needed)
0 18 * * * cd /path/to/oracle-x && /usr/bin/python3 main.py >> cronlog.txt 2>&1
3️⃣ Add your backtest:

cron
Copy
Edit
# Run backtest daily after the market close
0 19 * * * cd /path/to/oracle-x && /usr/bin/python3 backtest_tracker/backtest.py >> backtestlog.txt 2>&1
0 19 * * * cd /path/to/oracle-x && /usr/bin/python3 backtest_tracker/results_analyzer.py >> backtestlog.txt 2>&1
✅ That’s it — your clairvoyant pipeline:

Generates tomorrow’s tape → saves Playbook → backtests → self-corrects.

# Run after your backtest finishes
0 19 * * * cd /path/to/oracle-x && /usr/bin/python3 backtest_tracker/results_dashboard.py >> dashboardlog.txt 2>&1


🧩 Advanced: Expand It
✅ Connect real Twitter/X and Reddit sentiment scrapers
✅ Automate fetching charts from your broker API
✅ Pipe your playbook to a local vector DB for backtesting
✅ Build a local Streamlit dashboard to monitor signals overnight

🚀 Your Edge Starts Here
No more guesswork. No stale tips.
A daily, evolving, LLM-powered clairvoyant edge — 100% under your control.
Refine, plug in more tools, and test until it feels uncanny.

🪙 Disclaimer
This is not financial advice. This is a technical demo and research tool only.
Trade smart. Test everything. The market eats careless traders for breakfast.

## CLI Validation

Use the CLI utility to query each cataloged data point and paste results into `docs/DATA_VALIDATION_CHECKLIST.md`.

Examples:
- Quote
  - `python cli_validate.py quote --symbol AAPL`
- Market Data
  - `python cli_validate.py market_data --symbol AAPL --period 1y --interval 1d --preferred_sources twelve_data`
- Company Info
  - `python cli_validate.py company_info --symbol MSFT`
- News
  - `python cli_validate.py news --symbol AAPL --limit 5`
- Multiple Quotes
  - `python cli_validate.py multiple_quotes --symbols AAPL,MSFT,SPY`
- Financial Statements
  - `python cli_validate.py financial_statements --symbol AAPL`
- Sentiment (basic)
  - `python cli_validate.py sentiment --symbol TSLA`
- Advanced Sentiment
  - `python cli_validate.py advanced_sentiment --symbol TSLA`
- Market Breadth
  - `python cli_validate.py market_breadth`
- Sector Performance
  - `python cli_validate.py sector_performance`
- Comparison helper
  - `python cli_validate.py compare --value 195.23 --ref_value 196.5 --tolerance_pct 2.0`

Paste the raw JSON outputs into the “Execution Log” section of `docs/DATA_VALIDATION_CHECKLIST.md`, then update each scenario row with:
- Retrieved: source, timestamp, value summary
- Web Reference: a URL and value snapshot
- Pct Diff + Pass/Fail: use the compare helper to compute tolerance-based result
- Notes: anomalies, cache usage, or “skipped (missing key)” where applicable

## 🧠 Advanced Sentiment Enhancements (New)

The advanced sentiment pipeline aggregates multi-source textual signals before the ensemble (VADER + FinBERT + financial lexicon heuristics). Potential sources (when available): Reddit, Twitter/X, Yahoo Finance headlines, FinViz headlines, Yahoo News sentiment adapter sample texts, and Generic RSS feeds.

Key capabilities:
- Per‑source cap (env: `ADVANCED_SENTIMENT_MAX_PER_SOURCE`, default 300)
- Text truncation (256 chars) to control token cost
- Deduplication across sources
- Global truncation (3 × per‑source cap)
- Metadata: `raw_data['aggregated_counts']` with per-source counts + `total_unique`

Environment variables:
| Variable | Purpose | Default |
|----------|---------|---------|
| `ADVANCED_SENTIMENT_MAX_PER_SOURCE` | Max texts per source | `300` |
| `RSS_FEEDS` | Comma-separated RSS feed URLs | (empty) |
| `RSS_INCLUDE_ALL` | Include RSS headlines even without symbol match | `0` |

Example:
```bash
export ADVANCED_SENTIMENT_MAX_PER_SOURCE=120
export RSS_FEEDS="https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
export RSS_INCLUDE_ALL=1
python cli_validate.py advanced_sentiment --symbol TSLA
```

Inspect counts:
```bash
python -c "from data_feeds.data_feed_orchestrator import DataFeedOrchestrator; o=DataFeedOrchestrator(); adv=o.get_advanced_sentiment_data('AAPL'); import json; print(json.dumps(adv.raw_data.get('aggregated_counts',{}), indent=2))"
```

The FinViz ambiguous DataFrame truth-value path has been refactored to explicit sequential checks, eliminating prior warnings.