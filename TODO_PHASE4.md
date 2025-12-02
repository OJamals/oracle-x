# Phase 4: Optimization + Streamline oracle_engine + Final Fixes

## Core Goals
- Consolidate duplicates/redundancies: Merge oracle_engine/prompt_chain.py and prompt_chain_optimized.py into single optimized prompt_chain.py (remove duplicates, keep best from optimized version). Eliminate all 'prompt chain' duplicates across oracle_engine/.
- Centralize LLM invocations: Create oracle_engine/llm_dispatcher.py with caching, retries, config-driven models/providers.
- Streamline functions: Refactor oracle_engine/ modules (oracle_core.py, reasoning_engine.py) to remove redundant LLM/prompt logic. Inline short utils, extract shared helpers to oracle_engine/utils.py. Optimize prompt boosting with vector_db integration.
- Enhance structure/organization: Config-driven: Add config.yaml entries for news/sentiment sources. Integrate into data_feeds/ with toggles. Caches: Fully integrate/optimize all caches (conso_cache in data_feeds/cache/, vector_db). Add TTL/expiration, LRU eviction, Redis fallback. Update oracle_engine/ to use unified cache layer. Modularize: Group oracle_engine/ into subdirs (prompts/, chains/, dispatchers/, utils/). Ensure type hints, docstrings, async where possible.
- Remove unnecessary code: Delete dead imports/code (unused scrapers, old adapters). Prune verbose logging/debug prints. Static-analyze for unused vars/functions (pylint/mypy).
- Optimize performance: Batch LLM calls, parallelize data fetches (asyncio), reduce token usage in prompts. Vectorize sentiment/news processing.
- Quality gates: Ensure 100% pytest tests/ -v pass post-refactor. Add missing tests for new utils/caches. Lint with black/isort/pre-commit.

## Workflow
1. codebase_search for remaining duplicates ('prompt chain', 'llm call', 'fetch_' in oracle_engine/data_feeds).
2. Backup changes: git add -p, commit granularly (e.g., "consolidate prompt_chain", "unify LLM dispatcher").
3. Refactor iteratively: Edit files via editedExistingFile/readFile, verify with pytest.
4. Update todos via updateTodoList (mark completed, add new if needed).
5. Final: git status clean, push refactor-backup, summarize changes.

## Sequential Tasks

### 1. Search and Identify Duplicates
- [ ] Search 'prompt chain' in oracle_engine/ - found in agent.py, prompt_chain_optimized.py
- [ ] Search 'llm call' in oracle_engine/ - found in llm_providers.py, prompt_chain_optimized.py
- [ ] Search 'fetch_' in oracle_engine/ - many in async_data_fetcher.py
- [ ] Search 'fetch_' in data_feeds/ - many adapters
- [ ] Identify scattered LLM calls: call_llm, llm_chain.invoke, etc.

### 2. Consolidate Prompt Chain
- [ ] Merge prompt_chain.py and prompt_chain_optimized.py into single prompt_chain.py
- [ ] Keep optimization features from optimized version
- [ ] Remove duplicates, integrate best practices
- [ ] Update imports across codebase

### 3. Create Unified LLM Dispatcher
- [ ] Create oracle_engine/llm_dispatcher.py
- [ ] Centralize all LLM invocations
- [ ] Add caching, retries, config-driven models/providers
- [ ] Update all callers to use dispatcher

### 4. Streamline Oracle Engine Modules
- [ ] Refactor oracle_core.py, reasoning_engine.py - remove redundant LLM logic
- [ ] Inline short utils, extract shared helpers to oracle_engine/utils.py
- [ ] Optimize prompt boosting with vector_db integration (build_boosted_prompt/batch_build_boosted_prompts)

### 5. Enhance Structure and Organization
- [ ] Add config.yaml entries for news/sentiment sources with toggles
- [ ] Integrate toggles into data_feeds/
- [ ] Fully integrate caches: conso_cache, vector_db with TTL/LRU/Redis fallback
- [ ] Update oracle_engine/ to use unified cache layer
- [ ] Modularize oracle_engine/ into subdirs: prompts/, chains/, dispatchers/, utils/
- [ ] Add type hints, docstrings, async where possible

### 6. Remove Unnecessary Code
- [ ] Delete dead imports/code (unused scrapers, old adapters)
- [ ] Prune verbose logging/debug prints
- [ ] Static-analyze for unused vars/functions with pylint/mypy

### 7. Optimize Performance
- [ ] Batch LLM calls
- [ ] Parallelize data fetches with asyncio
- [ ] Reduce token usage in prompts
- [ ] Vectorize sentiment/news processing

### 8. Quality Gates
- [ ] Ensure 100% pytest tests/ -v pass
- [ ] Add missing tests for new utils/caches
- [ ] Lint with black/isort/pre-commit

### 9. Backup and Commit
- [ ] git add -p, commit granularly
- [ ] Final git status clean, push refactor-backup</content>
<parameter name="filePath">/Users/omar/Documents/Projects/oracle-x/TODO_PHASE4.md