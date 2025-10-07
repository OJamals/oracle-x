# 🗂️ ORACLE-X — Project Structure

---

## 🧠 `oracle_engine/`

| File | Purpose |
|------|---------|
| `agent.py` | The single entrypoint. Orchestrates the full Prompt Chain. |
| `prompt_chain.py` | Multi-step chain: pulls signals ➝ Vector recall ➝ adjusts scenario tree ➝ final Playbook. |
| `tools.py` | Your custom LLM tools — sentiment analyzer, chart interpreter, etc. |

---

## 🗃️ `vector_db/`

| File | Purpose |
|------|---------|
| `qdrant_store.py` | CRUD for storing & retrieving trade pattern vectors in local storage (JSON/pickle). |
| `prompt_booster.py` | Queries local vector storage for similar past scenarios to boost tomorrow's tape. |


**Note**: Vector storage now uses simple local files in `data/vector_store/` (no external DB required). Embeddings use OpenAI API from `OPENAI_API_BASE`.
---

## 📡 `data_feeds/`

| File | Purpose |
|------|---------|
| `market_internals.py` | Breadth, VIX, TRIN, up/down ratios. |
| `options_flow.py` | Unusual option sweeps (Unusual Whales, FlowAlgo). |
| `dark_pools.py` | Dark pool block trades (BlackBoxStocks, Cheddar Flow). |
| `sentiment.py` | Twitter/Reddit/Google Trends sentiment. |
| `earnings_calendar.py` | Earnings calendar feed (Finviz/Yahoo). |
| `anomaly_detector.py` | Z-score or custom outlier detection on price/volume. |

---

## 🗂️ `/signals/`

Daily raw feeds — versioned JSON. This is your data provenance.

---

## 🗂️ `/playbooks/`

Clairvoyant predictions. Each file is your “Tomorrow’s Tape.”  
Used in dashboards + backtest loops.

---

## 🏦 `/backtest_tracker/`

Tracks your hit/miss performance over time.  
Includes: `backtest.py`, `results_analyzer.py`, `results_dashboard.py`.

---

## 🖥️ `/dashboard/`

Your Streamlit app: Playbook visualizer + Operator Panel.  
`dashboard/app.py` shows:
- Playbook details
- Trade expanders
- Scenario tree charts
- System health status (Qdrant, Qwen3, Signals, Playbook timestamps)

---

## ✅ Key Principle

**`oracle_engine/` = brain**  
**`vector_db/` = memory**  
**`data_feeds/` = fresh signals**  
**`/signals/` = truth**  
**`/playbooks/` = predictions**  
**`/backtest_tracker/` = performance**  
**`/dashboard/` = cockpit**

---

Keep this folder structure clean. It’s the spine of your self-learning quant agent.
