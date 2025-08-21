# TODO Remediation Action Plan
========================================

## HIGH Priority (7 items)
### Medium Effort (1 items)
- [ ] **scripts/analysis/compare_pipelines.py:102** - Using generic method call since exact API unknown
### Large Effort (6 items)
- [ ] **backtest_tracker/comprehensive_backtest.py:360** - Implement parameter optimization
- [ ] **backtest_tracker/comprehensive_backtest.py:679** - Implement expanding window logic
- [ ] **backtest_tracker/comprehensive_backtest.py:685** - Implement rolling window logic
- [ ] **data_feeds/oracle_data_interface.py:161** - Implement real earnings date lookup
- [ ] **data_feeds/oracle_data_interface.py:182** - Implement with yfinance options data
- [ ] **.archive/agent_bundle_backup/data_feed_orchestrator.py:1434** - Implement issue tracking

## MEDIUM Priority (20 items)
### Medium Effort (20 items)
- [ ] **todo_analyzer.py:31** - , FIXME, HACK, etc.
- [ ] **todo_analyzer.py:219** - /FIXME Analysis Report")
- [ ] **todo_analyzer.py:267** - Remediation Action Plan")
- [ ] **main.py:60** - d pipeline imports (optional)
- [ ] **oracle_engine/ml_production_pipeline.py:139** - Full integration would require proper data and sentiment engines
- [ ] **oracle_engine/ml_production_pipeline.py:381** - Simplified signal generation for now
- [ ] **oracle_engine/prompt_optimization.py:461** - signal selection
- [ ] **oracle_engine/ml_model_manager.py:299** - Actual retraining would be triggered here
- [ ] **oracle_engine/ml_model_manager.py:354** - Full ensemble initialization would require:
- [ ] **oracle_engine/ml_model_manager.py:428** - This would use the ensemble engine when properly initialized
- [ ] **oracle_engine/ml_model_manager.py:476** - This would require updating the ensemble engine's internal models
- [ ] **data_feeds/oracle_data_interface.py:137** - Enhance with more sophisticated breadth analysis
- [ ] **data_feeds/finance_integration.py:319** - Cryptocurrencies may not be available in all versions
- [ ] **data_feeds/data_feed_orchestrator.py:103** - d utility normalization helpers (shared)
- [ ] **data_feeds/data_feed_orchestrator.py:646** - d field extraction with safe defaults
- [ ] **.archive/agent_bundle_backup/finance_integration.py:319** - Cryptocurrencies may not be available in all versions
- [ ] **.archive/backups/pipeline_originals/main_old.py:22** - d DataFeedOrchestrator with quality validation and caching
- [ ] **.archive/backups/pipeline_originals/main_original.py:22** - d DataFeedOrchestrator with quality validation and caching
- [ ] **README.md:41** - s
- [ ] **docs/data_endpoint_catalog.md:474** - s on Integration Strategy

## LOW Priority (9 items)
### Small Effort (6 items)
- [ ] **dashboard/app.py:101** - Rename this here and in `auto_generate_market_summary`
- [ ] **oracle_engine/prompt_chain.py:134** - Rename this here and in `extract_scenario_tree`
- [ ] **data_feeds/ticker_universe.py:29** - Rename this here and in `fetch_ticker_universe`
- [ ] **.archive/agent_bundle_backup/ticker_universe.py:29** - Rename this here and in `fetch_ticker_universe`
- [ ] **.archive/agent_bundle_backup/prompt_chain.py:134** - Rename this here and in `extract_scenario_tree`
- [ ] **.archive/agent_bundle_backup/reddit_sentiment.py:213** - Rename this here and in `fetch_reddit_sentiment`
### Medium Effort (3 items)
- [ ] **backtest_tracker/comprehensive_backtest.py:359** - strategy parameters on training data (placeholder for now)
- [ ] **tests/test_remaining_sources.py:184** - quantsumore has disrupted equity endpoints due to Yahoo Finance protections
- [ ] **tests/test_finance_libraries.py:460** - quarterly parameter not supported in current version
