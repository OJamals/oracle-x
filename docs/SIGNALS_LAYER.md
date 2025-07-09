# 📡 ORACLE-X — Signals Layer & Data Pipeline

This doc explains how your real-time data feeds work, what they connect to, and how they flow into your Prompt Chain to boost your clairvoyant Playbooks.

---

## ⚙️ **Signals Folder**

All raw daily scrapes are versioned in:  
`/signals/YYYY-MM-DD.json`

**Why?**  
✅ Audit your data quality  
✅ Backtest your Playbooks vs. real conditions  
✅ Spot which feeds are noisy, stale, or unreliable

---

## ✅ **Feeds Overview**

---

### 1️⃣ Market Internals

| Source       | Finviz Screener, Yahoo Finance, Alpha Vantage |
|--------------|-----------------------------------------------|
| Typical Data | Advancers, decliners, up/down volume, VIX, TRIN |
| Notes        | Yahoo is free for VIX & indices. Finviz works for breadth. |

**Sample Output:**
```json
{
  "breadth": {"advancers": 3200, "decliners": 1400, "up_volume": 2500000000, "down_volume": 1100000000},
  "vix": 17.5,
  "trin": 0.85
}
2️⃣ Options Flow
Source	Unusual Whales (paid), FlowAlgo (paid), or scrape Twitter bots like @unusual_whales
Typical Data	Big sweeps, unusual OI changes, block trades
Notes	For live data, a paid feed is recommended for reliability.

Sample Output:

json
Copy
Edit
{
  "unusual_sweeps": [
    {"ticker": "AAPL", "direction": "Call", "strike": 200, "volume": 10000},
    {"ticker": "TSLA", "direction": "Put", "strike": 750, "volume": 8000}
  ]
}
3️⃣ Dark Pools
Source	BlackBoxStocks, Cheddar Flow, or scrape broker dark pool prints
Typical Data	Large off-exchange block prints; hidden directional bets
Notes	Paid feeds show real-time prints.

Sample Output:

json
Copy
Edit
{
  "dark_pools": [
    {"ticker": "AAPL", "block_size": 500000, "price": 192.50},
    {"ticker": "TSLA", "block_size": 350000, "price": 805.00}
  ]
}
4️⃣ Sentiment Data
Source	Twitter (snscrape or API v2), Reddit (PRAW), Google Trends (pytrends)
Typical Data	Bullish/bearish keyword counts, trending tickers
Notes	Use snscrape for free scraping, or small paid APIs for real-time.

Sample Output:

json
Copy
Edit
{
  "TSLA": {"twitter": "Positive", "reddit": "Bullish"},
  "NVDA": {"twitter": "Mixed", "reddit": "Neutral"}
}
5️⃣ Earnings Calendar
Source	Finviz, Yahoo Finance Earnings API
Typical Data	Upcoming earnings dates & estimates
Notes	Finviz is free — scrape their HTML or use Yahoo’s free endpoints.

Sample Output:

json
Copy
Edit
[
  {"ticker": "AAPL", "date": "2025-07-10", "estimate": 1.32},
  {"ticker": "TSLA", "date": "2025-07-12", "estimate": 2.15}
]
6️⃣ Anomaly Detector
Source	Your own local signal logic (e.g., Z-score outlier detection)
Typical Data	Time series anomalies: sudden price/volume spikes
Notes	Use pandas, scipy, or any open-source TS anomaly lib.

Sample Output:

json
Copy
Edit
[
  {"ticker": "TSLA", "timestamp": "2025-07-09 14:30", "zscore": 3.2, "note": "Volume spike"}
]
✅ Free vs. Paid Cheat Sheet
Feed Type	Free	Paid (optional)
Market Internals	Finviz, Yahoo scraping	Alpha Vantage Pro
Options Flow	Scrape unusual flow Twitter bots	Unusual Whales, FlowAlgo
Dark Pools	Broker site scraping	BlackBoxStocks, Cheddar Flow
Sentiment	snscrape, PRAW, pytrends	Brandwatch, SocialSentiment APIs
Earnings	Finviz, Yahoo	Benzinga API
Anomaly Logic	Local pandas/scipy	N/A — always local

✅ Best Practices
✔️ Save each day’s signals in /signals/ before you run your agent.
✔️ If any feed fails, log an empty {} — don’t break the pipeline.
✔️ Rotate your free API keys regularly. Store them in config/api_keys.yaml.
✔️ Never hardcode creds in the scrapers.
✔️ Use proxies or delay loops for Twitter & Google Trends scraping to avoid bans.
✔️ Version every feed: they’re your raw “truth” for backtesting your Playbooks.

🕒 Example Cron Job
Here’s how to automate your full loop daily:

cron
Copy
Edit
# Run the signals scraper every day at 5:30 PM
30 17 * * * cd /path/to/oracle-x && /usr/bin/python3 signals_runner.py >> signalslog.txt 2>&1

# Run the main agent pipeline at 6:00 PM
0 18 * * * cd /path/to/oracle-x && /usr/bin/python3 main.py >> mainlog.txt 2>&1

# Run your backtest tracker at 6:30 PM
30 18 * * * cd /path/to/oracle-x && /usr/bin/python3 backtest_tracker/backtest.py >> backtestlog.txt 2>&1
Check *.txt logs each morning for errors.

✅ Your Alpha Feeds — Evolving Daily
These signals are your lifeblood.
Start with free scrapes ➝ swap in premium data ➝ refine your Prompt Chain ➝ backtest ➝ improve your edge.
Keep it open, versioned, and local.

This is how you build a quant-grade prediction loop — clairvoyant, auditable, and evolving.

🔮📈✨

yaml
Copy
Edit

---

## ✅ **Why This Is Solid**

✔️ *Beautifully structured*: easy to skim for new devs  
✔️ *Exact JSON examples*: you can test your feeds one at a time  
✔️ *Practical Cron*: run daily, logs versioned  
✔️ *Local-first*: open-source, no vendor lock  
✔️ *Clear next steps*: plug in premium feeds when you’re ready
