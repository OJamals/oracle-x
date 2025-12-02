# Quick Migration Guide for Oracle-X Refactoring

## TL;DR - What Changed?

1. **Configuration**: Use `from core.config import config` instead of old modules
2. **Database paths**: Now in `data/databases/`
3. **Log files**: Now in `logs/`
4. **Scripts**: Organized in `scripts/validation/`, `scripts/check/`, etc.
5. **Tests**: All in `tests/` with subdirectories

## Immediate Actions Required

### None! 🎉
Everything still works with backward compatibility. You can migrate gradually.

## Recommended Migration (When You Have Time)

### 1. Update Configuration Imports

**Old (still works, shows warning):**
```python
from env_config import get_openai_model, load_config
from config_manager import DatabaseConfig, get_cache_db_path
from config_validator import ConfigValidator
```

**New (recommended):**
```python
from core.config import config, DatabaseConfig, ConfigValidator

# Access configuration
model = config.model.openai_model
db_path = config.database.get_full_path('accounts')
cache_path = config.database.get_full_path('cache')

# Load config dict (for compatibility)
config_dict = config.to_dict()
```

### 2. Update Database Paths

**Old:**
```python
db_path = "accounts.db"
cache_db = "cache.db"
```

**New:**
```python
from core.config import config

db_path = config.database.get_full_path('accounts')
cache_path = config.database.get_full_path('cache')

# Or use string paths
db_path = str(config.database.get_full_path('accounts'))
```

### 3. Update Utility Imports

**Old:**
```python
from common_utils import setup_logging, get_project_root
```

**New:**
```python
from utils.common import setup_logging, get_project_root
```

### 4. Environment Variables (Optional)

You can now configure paths via environment variables:

```bash
# .env or shell
export ACCOUNTS_DB_PATH="data/databases/accounts.db"
export MODEL_MONITORING_DB_PATH="data/databases/model_monitoring.db"
export CACHE_DB_PATH="data/cache/cache.db"
export LOG_DIR="logs"
```

## What's Deprecated (But Still Works)

These modules show deprecation warnings but are fully functional:
- `env_config.py` → Use `core.config` instead
- `config_manager.py` → Use `core.config` instead
- `config_validator.py` → Use `core.config` instead
- `common_utils.py` (in root) → Use `utils.common` instead

## Configuration Examples

### Before & After

#### Getting Model Configuration
```python
# BEFORE
from env_config import get_openai_model, get_fallback_models
model = get_openai_model()
fallbacks = get_fallback_models()

# AFTER
from core.config import config
model = config.model.openai_model
fallbacks = config.model.fallback_models
```

#### Getting Database Paths
```python
# BEFORE
from config_manager import get_accounts_db_path
db_path = get_accounts_db_path()

# AFTER
from core.config import config
db_path = config.database.get_full_path('accounts')
# Or: db_path = config.database.get_accounts_db_path()  # backward compat
```

#### Validating Configuration
```python
# BEFORE
from config_validator import ConfigValidator
validator = ConfigValidator()
result = validator.validate()

# AFTER
from core.config import config, ConfigValidator
validator = ConfigValidator(config)
result = validator.validate()
```

#### Accessing API Keys
```python
# BEFORE
from config_manager import get_finnhub_api_key
api_key = get_finnhub_api_key()

# AFTER
from core.config import config
api_key = config.data_feeds.finnhub_api_key
```

## Testing Your Migration

### Run Tests
```bash
# All tests should pass
pytest

# Check for deprecation warnings
pytest -W default::DeprecationWarning
```

### Verify Configuration
```python
from core.config import config, ConfigValidator

# Check configuration loads
print(f"Model: {config.model.openai_model}")
print(f"DB Path: {config.database.get_full_path('accounts')}")

# Validate configuration
validator = ConfigValidator(config)
result = validator.validate()
if not result.is_valid:
    print("Errors:", result.errors)
    print("Warnings:", result.warnings)
```

## Common Patterns

### Pattern 1: Database Connection
```python
# BEFORE
import sqlite3
conn = sqlite3.connect("accounts.db")

# AFTER
import sqlite3
from core.config import config
conn = sqlite3.connect(config.database.get_full_path('accounts'))
```

### Pattern 2: Logging Setup
```python
# BEFORE
from common_utils import setup_logging
logger = setup_logging("my_module")

# AFTER
from utils.common import setup_logging
logger = setup_logging("my_module")
```

### Pattern 3: Configuration Access
```python
# BEFORE
from env_config import load_config
config = load_config()
api_key = config['OPENAI_API_KEY']

# AFTER
from core.config import config
api_key = config.model.openai_api_key
# Or for dict access: config_dict = config.to_dict()
```

## Troubleshooting

### "Module not found: env_config"
- **Solution**: The module exists but may have import issues. Check that you're in the project root.

### "Deprecation warnings everywhere"
- **Expected**: These warn you about old imports. Migrate when convenient.
- **To suppress**: `import warnings; warnings.filterwarnings('ignore', category=DeprecationWarning)`

### "Database path not found"
- **Solution**: The new structure creates parent directories automatically. Ensure you're using `config.database.get_full_path()`.

### "Tests failing after migration"
- **Check**: Import paths
- **Check**: Database paths
- **Solution**: Most tests should pass without modification due to backward compatibility.

## Timeline Recommendations

### Phase 1 (Optional - Anytime)
- Update configuration imports in new code
- Use `core.config` for new features

### Phase 2 (Optional - When Convenient)
- Migrate existing files gradually
- Update one module at a time
- Run tests after each change

### Phase 3 (Future - When Ready)
- Remove deprecated modules
- Remove backward compatibility shims
- Final cleanup

## Questions?

Check these resources:
- `docs/REFACTORING_SUMMARY_2025_10_07.md` - Complete refactoring details
- `docs/REFACTORING_BEFORE_AFTER.md` - Visual comparison
- `docs/STRUCTURE_REFACTORED.md` - New structure overview
- `.github/copilot-instructions.md` - Updated architecture guide

## Benefits You'll Get

- ✅ Type-safe configuration access
- ✅ Better IDE autocomplete
- ✅ Clearer code organization
- ✅ Built-in validation
- ✅ Hierarchical configuration
- ✅ Environment variable support
- ✅ Better error messages

---

**Remember**: Migration is optional and can be done gradually. All old code continues to work!
