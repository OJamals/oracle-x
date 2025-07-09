# ⚡️ ORACLE-X — CLI Cheatsheet

Everything you need to run your clairvoyant pipeline, test modules, and fix common issues — all from your terminal.

---

## ✅ 1️⃣ 📡 Run Signals Feeds

Pull today’s raw market data:
```bash
python signals_runner.py
Output: /signals/YYYY-MM-DD.json

✅ 2️⃣ 🔮 Generate the Playbook
Run the full Prompt Chain:

bash
Copy
Edit
python main.py
Output: /playbooks/YYYY-MM-DD.json + vectors stored in Qdrant.

✅ 3️⃣ 📈 Backtest It
Check yesterday’s predictions:

bash
Copy
Edit
python backtest_tracker/backtest.py
python backtest_tracker/results_analyzer.py
✅ 4️⃣ 🗃️ Streamlit Dashboard
Run your full Operator & Playbook Dashboard:

bash
Copy
Edit
streamlit run dashboard/app.py
✅ 5️⃣ 🕒 Check Cron Jobs
View your active crontab:

bash
Copy
Edit
crontab -l
Edit it:

bash
Copy
Edit
crontab -e
Example Cron Block:

cron
Copy
Edit
30 17 * * * cd /path/to/oracle-x && /usr/bin/python3 signals_runner.py >> signalslog.txt 2>&1
0 18 * * * cd /path/to/oracle-x && /usr/bin/python3 main.py >> mainlog.txt 2>&1
30 18 * * * cd /path/to/oracle-x && /usr/bin/python3 backtest_tracker/backtest.py >> backtestlog.txt 2>&1
✅ 6️⃣ 🧩 Check Qdrant & Qwen3
Check Qdrant:

bash
Copy
Edit
curl http://localhost:6333/collections
Check Qwen3 Embedding Server:

bash
Copy
Edit
curl http://localhost:8000/v1/models
If they fail, restart the container or service.

✅ 7️⃣ 🐳 Docker Compose (Optional)
If you run Qdrant & Qwen3 via Docker Compose:

bash
Copy
Edit
docker-compose up -d
docker-compose ps
docker-compose logs qdrant
docker-compose logs qwen3
docker-compose down
Example docker-compose.yml:

yaml
Copy
Edit
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
  qwen3:
    image: your-qwen3-embedding-server
    ports:
      - "8000:8000"
✅ 8️⃣ 🗂️ File Debug Tips
Signals not saving? → Check /signals/ exists.

Playbook JSON error? → Inspect output for raw_output fallback.

Vectors missing? → Check store_trade_vector() logs.

✅ 🔄 Rebuild / Restart Tips
If anything breaks:

bash
Copy
Edit
# Restart Qdrant container
docker restart qdrant

# Restart Qwen3 embedding server
docker restart qwen3

# Re-run your signals + agent
python signals_runner.py
python main.py
✅ 9️⃣ 🏁 Quick Testing
Test a single feed:

bash
Copy
Edit
python -m data_feeds.market_internals
Test your Prompt Booster alone:

bash
Copy
Edit
python -m vector_db.prompt_booster
⚡️ Final Golden Rule
Signals ➝ Playbook ➝ Vectors ➝ Backtest ➝ Dashboard.

Keep your logs/*.txt fresh — they’re your truth window.

Stay clairvoyant.