# 🧙‍♂️ ORACLE-X — Daily Operator SOP

This is your bulletproof daily playbook for running ORACLE-X.

---

## ✅ Daily Checklist

1️⃣ **Run `signals_runner.py`**
- Pulls market internals, options flow, dark pools, sentiment, earnings.
- Saves to `/signals/YYYY-MM-DD.json`.

2️⃣ **Run `main.py`**
- Loads signals + chart screenshot.
- Runs Prompt Chain ➝ Qdrant recall ➝ scenario tree ➝ final Playbook.
- Saves to `/playbooks/YYYY-MM-DD.json`.
- Stores vectors in Qdrant for recall.

3️⃣ **Run Backtest**
- `python backtest_tracker/backtest.py`
- `python backtest_tracker/results_analyzer.py`
- Verifies predictions vs. real market moves.

4️⃣ **Check Dashboard**
- `streamlit run dashboard/app.py`
- Confirm Operator Panel: Qdrant ✅ Qwen3 ✅ Latest signals ✅ Latest Playbook ✅

---

## ✅ Weekly Maintenance

✔️ Rotate free API keys if needed.
✔️ Check feeds for scraping blocks or API changes.
✔️ Back up `/signals/` & `/playbooks/` for training data.
✔️ Monitor backtest hit rate for prompt tweaks.

---

## 🕒 Cron Template

```cron
30 17 * * * cd /path/to/oracle-x && /usr/bin/python3 signals_runner.py >> signalslog.txt 2>&1
0 18 * * * cd /path/to/oracle-x && /usr/bin/python3 main.py >> mainlog.txt 2>&1
30 18 * * * cd /path/to/oracle-x && /usr/bin/python3 backtest_tracker/backtest.py >> backtestlog.txt 2>&1
