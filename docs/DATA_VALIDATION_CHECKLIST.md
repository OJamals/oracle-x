# Data Validation Checklist

Purpose: Use this checklist to manually validate orchestrated data outputs against a simple web reference. Paste CLI JSON under "Execution Log" and summarize results in the fields below. Mark each line item as [ ] Pending, [x] Pass, or [x] Fail with notes.

Tickers set to validate: AAPL, MSFT, SPY, TSLA, JPM, XOM, BABA, MU, PLUG/RIOT, INVALID1234

Conventions
- Retrieved: Source / Timestamp / Value summary
- Web Ref: Link and captured value
- Diff: Absolute and Percent; mark Pass/Fail using tolerance rule you choose (e.g., <=2%)
- Notes: Brief rationale, anomalies, cache indicators, or missing key skips
- For multi-row outputs (market data, breadth, sector), summarize the last-row or key aggregates

How to use
1) Run a CLI command from the list below.
2) Paste the JSON output into the Execution Log section.
3) For each scenario checklist row, fill:
   - Retrieved: source, iso timestamp, key value(s)
   - Web Ref: paste URL and reference value scraped/seen
   - Diff/Result: compute pct diff (or use the CLI compare helper) and mark Pass/Fail
   - Notes: anomalies, missing keys, adapter skips, limitations

Tolerance helper
- Use the generic compare helper:
  - python cli_validate.py compare --value <retrieved> --ref_value <web> --tolerance_pct 2.0

Sections

Quote
- Command: python cli_validate.py quote --symbol <SYM> [--preferred_sources yfinance|twelve_data]
- Fields: price, change_percent
- Scenarios:
  - [ ] AAPL | Retrieved: ____ | Web Ref: ____ | Diff: ____% → Pass/Fail | Notes: ____
  - [ ] MSFT | Retrieved: ____ | Web Ref: ____ | Diff: ____% → Pass/Fail | Notes: ____
  - [ ] SPY  | Retrieved: ____ | Web Ref: ____ | Diff: ____% → Pass/Fail | Notes: ____
  - [ ] TSLA | Retrieved: ____ | Web Ref: ____ | Diff: ____% → Pass/Fail | Notes: ____
  - [ ] JPM  | Retrieved: ____ | Web Ref: ____ | Diff: ____% → Pass/Fail | Notes: ____
  - [ ] XOM  | Retrieved: ____ | Web Ref: ____ | Diff: ____% → Pass/Fail | Notes: ____
  - [ ] BABA | Retrieved: ____ | Web Ref: ____ | Diff: ____% → Pass/Fail | Notes: ____
  - [ ] MU   | Retrieved: ____ | Web Ref: ____ | Diff: ____% → Pass/Fail | Notes: ____
  - [ ] PLUG | Retrieved: ____ | Web Ref: ____ | Diff: ____% → Pass/Fail | Notes: ____
  - [ ] RIOT | Retrieved: ____ | Web Ref: ____ | Diff: ____% → Pass/Fail | Notes: ____
  - [ ] INVALID1234 | Retrieved: ____ | Web Ref: N/A | Result: Expected failure handled | Notes: ____

Market Data
- Command: python cli_validate.py market_data --symbol <SYM> --period 1y --interval 1d [--preferred_sources yfinance|twelve_data]
- Fields: timeframe, last row OHLCV summary
- Scenarios:
  - [ ] AAPL | Retrieved: ____ | Web Ref: ____ | Diff Close: ____% → Pass/Fail | Notes: ____
  - [ ] MSFT | Retrieved: ____ | Web Ref: ____ | Diff Close: ____% → Pass/Fail | Notes: ____
  - [ ] SPY  | Retrieved: ____ | Web Ref: ____ | Diff Close: ____% → Pass/Fail | Notes: ____
  - [ ] TSLA | Retrieved: ____ | Web Ref: ____ | Diff Close: ____% → Pass/Fail | Notes: ____
  - [ ] JPM  | Retrieved: ____ | Web Ref: ____ | Diff Close: ____% → Pass/Fail | Notes: ____
  - [ ] XOM  | Retrieved: ____ | Web Ref: ____ | Diff Close: ____% → Pass/Fail | Notes: ____
  - [ ] BABA | Retrieved: ____ | Web Ref: ____ | Diff Close: ____% → Pass/Fail | Notes: ____
  - [ ] MU   | Retrieved: ____ | Web Ref: ____ | Diff Close: ____% → Pass/Fail | Notes: ____
  - [ ] PLUG | Retrieved: ____ | Web Ref: ____ | Diff Close: ____% → Pass/Fail | Notes: ____
  - [ ] RIOT | Retrieved: ____ | Web Ref: ____ | Diff Close: ____% → Pass/Fail | Notes: ____
  - [ ] INVALID1234 | Retrieved: ____ | Web Ref: N/A | Result: Expected failure handled | Notes: ____

Company Info
- Command: python cli_validate.py company_info --symbol <SYM>
- Fields: name, sector, market_cap
- Scenarios:
  - [ ] AAPL | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] MSFT | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] SPY  | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] TSLA | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] JPM  | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] XOM  | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] BABA | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] MU   | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] PLUG | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] RIOT | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] INVALID1234 | Retrieved: ____ | Web Ref: N/A | Result: Expected failure handled | Notes: ____

News
- Command: python cli_validate.py news --symbol <SYM> --limit 5
- Fields: first N titles + published timestamps
- Scenarios:
  - [ ] AAPL | Retrieved: ____ | Web Ref: ____ | Titles aligned? Pass/Fail | Notes: ____
  - [ ] MSFT | Retrieved: ____ | Web Ref: ____ | Titles aligned? Pass/Fail | Notes: ____
  - [ ] TSLA | Retrieved: ____ | Web Ref: ____ | Titles aligned? Pass/Fail | Notes: ____
  - [ ] INVALID1234 | Retrieved: ____ | Result: Expected empty/handled | Notes: ____

Multiple Quotes
- Command: python cli_validate.py multiple_quotes --symbols AAPL,MSFT,SPY
- Fields: per-symbol: price, change_percent
- Scenarios:
  - [ ] AAPL,MSFT,SPY | Retrieved: ____ | Web Ref: ____ | Diff within tolerance? Pass/Fail | Notes: ____

Financial Statements
- Command: python cli_validate.py financial_statements --symbol <SYM>
- Fields: shapes of income_statement, balance_sheet, cash_flow (if available)
- Scenarios:
  - [ ] AAPL | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] MSFT | Retrieved: ____ | Web Ref: ____ | Pass/Fail | Notes: ____
  - [ ] INVALID1234 | Retrieved: ____ | Expected empty/handled | Notes: ____

Sentiment (Reddit/Twitter)
- Command: python cli_validate.py sentiment --symbol <SYM>
- Fields: per-source: score, confidence, sample_size
- Scenarios:
  - [ ] TSLA | Retrieved: ____ | Web Ref: ____ | Direction match? Pass/Fail | Notes: ____
  - [ ] AAPL | Retrieved: ____ | Web Ref: ____ | Direction match? Pass/Fail | Notes: ____
  - [ ] INVALID1234 | Retrieved: ____ | Expected skip/handled | Notes: ____

Advanced Sentiment
- Command: python cli_validate.py advanced_sentiment --symbol <SYM>
- Fields: score, confidence, sample_size
- Scenarios:
  - [ ] TSLA | Retrieved: ____ | Web Ref: ____ | Direction/strength reasonable? Pass/Fail | Notes: ____
  - [ ] AAPL | Retrieved: ____ | Web Ref: ____ | Direction/strength reasonable? Pass/Fail | Notes: ____
  - [ ] INVALID1234 | Retrieved: ____ | Expected skip/handled | Notes: ____

Market Breadth
- Command: python cli_validate.py market_breadth
- Fields: advancers, decliners, unchanged, put_call_ratio (if present)
- Scenarios:
  - [ ] Retrieved: ____ | Web Ref: ____ | Diff(s): ____ | Pass/Fail | Notes: ____

Sector Performance
- Command: python cli_validate.py sector_performance
- Fields: brief summary rows (perf_1d, perf_1w, perf_1m, perf_ytd)
- Scenarios:
  - [ ] Retrieved: ____ | Web Ref: ____ | Trend alignment? Pass/Fail | Notes: ____

Execution Log
Paste raw CLI outputs here for traceability.
