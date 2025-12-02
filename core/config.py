"""
⚙️ ORACLE-X Unified Configuration System

Consolidated configuration management for the ORACLE-X platform.
Merges functionality from env_config.py, config_manager.py, and config_validator.py.

Features:
- Hierarchical configuration loading (env vars → config files → defaults)
- Type-safe configuration access with validation
- Environment-specific configurations (dev, test, prod)
- Database path management with new data/ structure
- API key and model configurations
- Comprehensive validation with clear error messages

Usage:
    from core.config import config, ConfigValidator

    # Access configuration
    api_key = config.model.openai_api_key
    db_path = config.database.get_full_path('accounts')

    # Validate configuration
    validator = ConfigValidator()
    result = validator.validate()
    if not result.is_valid:
        print(result.errors)
"""

from __future__ import annotations
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Optional dotenv support
try:
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv("config/optimization.env")
    load_dotenv("config/rss_feeds_config.env")
except Exception:
    pass


# ========================== ENUMS ==========================


class Environment(Enum):
    """Environment types"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class ConfigValidationLevel(Enum):
    """Configuration validation levels"""

    STRICT = "strict"  # All required configs must be present
    PERMISSIVE = "permissive"  # Allow missing configs with warnings
    MINIMAL = "minimal"  # Only validate critical configs


# ========================== HELPER FUNCTIONS ==========================


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


# ========================== CONFIGURATION DATACLASSES ==========================


@dataclass
class DatabaseConfig:
    """Database configuration settings with new data/ structure"""

    accounts_db: str = "data/databases/accounts.db"
    model_monitoring_db: str = "data/databases/model_monitoring.db"
    prompt_optimization_db: str = "data/databases/prompt_optimization.db"
    cache_db: str = "data/cache/cache.db"

    def __post_init__(self):
        """Override from environment variables"""
        self.accounts_db = os.environ.get("ACCOUNTS_DB_PATH", self.accounts_db)
        self.model_monitoring_db = os.environ.get(
            "MODEL_MONITORING_DB_PATH", self.model_monitoring_db
        )
        self.prompt_optimization_db = os.environ.get(
            "PROMPT_OPTIMIZATION_DB_PATH", self.prompt_optimization_db
        )
        self.cache_db = os.environ.get("CACHE_DB_PATH", self.cache_db)

    def get_full_path(self, db_name: str) -> Path:
        """Get full path for database file

        Args:
            db_name: Name of database (accounts, model_monitoring, prompt_optimization, cache)

        Returns:
            Full path to database file
        """
        path_attr = f"{db_name}_db"
        if hasattr(self, path_attr):
            relative_path = getattr(self, path_attr)
            full_path = get_project_root() / relative_path
            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            return full_path
        raise ValueError(f"Unknown database: {db_name}")

    # Backward compatibility getters
    def get_accounts_db_path(self) -> str:
        return str(self.get_full_path("accounts"))

    def get_model_monitoring_db_path(self) -> str:
        return str(self.get_full_path("model_monitoring"))

    def get_prompt_optimization_db_path(self) -> str:
        return str(self.get_full_path("prompt_optimization"))

    def get_cache_db_path(self) -> str:
        return str(self.get_full_path("cache"))


@dataclass
class ModelConfig:
    """Model configuration settings"""

    openai_model: str = "gpt-4o"
    embedding_model: str = "qwen3-embedding"
    fallback_models: List[str] = field(
        default_factory=lambda: ["gpt-4o", "o4-mini", "o3-mini"]
    )
    openai_api_base: Optional[str] = None
    embedding_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None

    def __post_init__(self):
        """Override from environment variables"""
        self.openai_model = os.environ.get("OPENAI_MODEL", self.openai_model)
        self.embedding_model = os.environ.get("EMBEDDING_MODEL", self.embedding_model)
        self.openai_api_base = os.environ.get("OPENAI_API_BASE", self.openai_api_base)
        # Use OPENAI_API_BASE for embeddings by default
        # This allows the same API endpoint to be used for both LLM and embeddings
        self.embedding_api_base = self.openai_api_base
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", self.openai_api_key)

        # Parse fallback models from comma-separated env var
        fallback_env = os.environ.get("FALLBACK_MODELS", "")
        if fallback_env:
            self.fallback_models = [
                m.strip() for m in fallback_env.split(",") if m.strip()
            ]

    # Backward compatibility getters
    def get_openai_model(self) -> str:
        return self.openai_model

    def get_non_embedding_model(self) -> str:
        return self.openai_model

    def get_embedding_model(self) -> str:
        return self.embedding_model

    def get_openai_api_base(self) -> Optional[str]:
        return self.openai_api_base

    def get_embedding_api_base(self) -> Optional[str]:
        return self.embedding_api_base

    def get_fallback_models(self) -> List[str]:
        return self.fallback_models


@dataclass
class DataFeedConfig:
    """Data feed configuration settings"""

    # Primary API Keys
    financial_modeling_prep_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    alphavantage_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None

    # Social Media APIs
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None

    # Optional API Keys
    iex_api_key: Optional[str] = None
    tradingeconomics_api_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    newsapi_api_key: Optional[str] = None
    google_news_api_key: Optional[str] = None

    # Configuration Settings
    cache_ttl_minutes: int = 5
    max_retries: int = 3
    request_timeout: int = 30
    api_rate_limit_delay: float = 1.0
    enable_api_fallback: bool = True

    def __post_init__(self):
        """Override from environment variables"""
        self.financial_modeling_prep_api_key = (
            os.environ.get("FINANCIALMODELINGPREP_API_KEY")
            or os.environ.get("FMP_API_KEY")
            or self.financial_modeling_prep_api_key
        )
        self.finnhub_api_key = os.environ.get("FINNHUB_API_KEY", self.finnhub_api_key)
        self.alphavantage_api_key = os.environ.get(
            "ALPHAVANTAGE_API_KEY", self.alphavantage_api_key
        )
        self.twitter_bearer_token = os.environ.get(
            "TWITTER_BEARER_TOKEN", self.twitter_bearer_token
        )
        self.reddit_client_id = os.environ.get(
            "REDDIT_CLIENT_ID", self.reddit_client_id
        )
        self.reddit_client_secret = os.environ.get(
            "REDDIT_CLIENT_SECRET", self.reddit_client_secret
        )
        self.iex_api_key = os.environ.get("IEX_API_KEY", self.iex_api_key)
        self.tradingeconomics_api_key = os.environ.get(
            "TRADINGECONOMICS_API_KEY", self.tradingeconomics_api_key
        )
        self.fred_api_key = os.environ.get("FRED_API_KEY", self.fred_api_key)
        self.newsapi_api_key = os.environ.get("NEWSAPI_API_KEY", self.newsapi_api_key)
        self.google_news_api_key = os.environ.get(
            "GOOGLE_NEWS_API_KEY", self.google_news_api_key
        )


@dataclass
class VectorDBConfig:
    """Vector database configuration - Local ChromaDB"""

    storage_path: str = "data/vector_db"
    collection_name: str = "oraclex_trades"

    def __post_init__(self):
        """Override from environment variables"""
        self.storage_path = os.environ.get("VECTOR_DB_PATH", self.storage_path)
        self.collection_name = os.environ.get(
            "VECTOR_DB_COLLECTION", self.collection_name
        )

    def get_full_path(self) -> Path:
        """Get full path for vector database storage"""
        full_path = get_project_root() / self.storage_path
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path


@dataclass
class OptimizationConfig:
    """Optimization system configuration"""

    enabled: bool = True
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    max_generations: int = 50
    performance_window_days: int = 7

    def __post_init__(self):
        """Override from environment variables"""
        self.enabled = (
            os.environ.get("OPTIMIZATION_ENABLED", str(self.enabled)).lower() == "true"
        )
        self.population_size = int(
            os.environ.get("OPTIMIZATION_POPULATION_SIZE", self.population_size)
        )
        self.mutation_rate = float(
            os.environ.get("OPTIMIZATION_MUTATION_RATE", self.mutation_rate)
        )


@dataclass
class SystemConfig:
    """Complete system configuration"""

    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = False
    log_level: str = "INFO"
    log_dir: str = "logs"

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data_feeds: DataFeedConfig = field(default_factory=DataFeedConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def __post_init__(self):
        """Override from environment variables"""
        env_str = os.environ.get("ENVIRONMENT", "development")
        self.environment = Environment(env_str)
        self.debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
        self.log_level = os.environ.get("LOG_LEVEL", self.log_level)
        self.log_dir = os.environ.get("LOG_DIR", self.log_dir)

        # Ensure log directory exists
        log_path = get_project_root() / self.log_dir
        log_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for compatibility"""
        return {
            "OPENAI_API_KEY": self.model.openai_api_key,
            "OPENAI_MODEL": self.model.openai_model,
            "EMBEDDING_MODEL": self.model.embedding_model,
            "OPENAI_API_BASE": self.model.openai_api_base,
            "EMBEDDING_API_BASE": self.model.embedding_api_base,
            "FALLBACK_MODELS": self.model.fallback_models,
            "VECTOR_DB_PATH": self.vector_db.storage_path,
            "VECTOR_DB_COLLECTION": self.vector_db.collection_name,
            "ACCOUNTS_DB_PATH": self.database.accounts_db,
            "MODEL_MONITORING_DB_PATH": self.database.model_monitoring_db,
            "PROMPT_OPTIMIZATION_DB_PATH": self.database.prompt_optimization_db,
            "CACHE_DB_PATH": self.database.cache_db,
            "LOG_DIR": self.log_dir,
            "LOG_LEVEL": self.log_level,
            "DEBUG": self.debug_mode,
            "ENVIRONMENT": self.environment.value,
        }


# ========================== VALIDATION ==========================


@dataclass
class ConfigValidationResult:
    """Result of configuration validation"""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missing_configs: List[str]
    invalid_configs: List[str]


class ConfigValidator:
    """Validates ORACLE-X configuration settings"""

    # Required configuration variables
    REQUIRED_CONFIGS = {
        "OPENAI_API_KEY": {
            "description": "OpenAI API key for LLM operations",
            "validation": lambda x: x and len(x) > 10,
            "error_msg": "Must be a valid OpenAI API key (sk-...)",
        }
    }

    # Optional but recommended configurations
    RECOMMENDED_CONFIGS = {
        "OPENAI_MODEL": {
            "description": "Preferred OpenAI model",
            "default": "gpt-4o",
            "validation": lambda x: x
            in ["gpt-4.1", "gpt-4o", "o4-mini", "o3-mini", "gpt-4o-mini"],
            "error_msg": "Should be a supported OpenAI model",
        },
        "VECTOR_DB_PATH": {
            "description": "Local vector database storage path",
            "default": "data/vector_db",
            "validation": lambda x: x and len(x) > 0,
            "error_msg": "Must be a valid path",
        },
    }

    def __init__(
        self,
        config: SystemConfig,
        level: ConfigValidationLevel = ConfigValidationLevel.PERMISSIVE,
    ):
        self.config = config
        self.level = level

    def validate(self) -> ConfigValidationResult:
        """Validate configuration settings

        Returns:
            ConfigValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        missing_configs = []
        invalid_configs = []

        # Validate required configs
        for key, spec in self.REQUIRED_CONFIGS.items():
            value = self._get_config_value(key)
            if not value:
                missing_configs.append(key)
                errors.append(f"Missing required config: {key} - {spec['description']}")
            elif not spec["validation"](value):
                invalid_configs.append(key)
                errors.append(f"Invalid config: {key} - {spec['error_msg']}")

        # Validate recommended configs (warnings only)
        for key, spec in self.RECOMMENDED_CONFIGS.items():
            value = self._get_config_value(key)
            if not value:
                warnings.append(
                    f"Recommended config missing: {key} - {spec['description']}"
                )
            elif "validation" in spec and not spec["validation"](value):
                warnings.append(f"Config may be invalid: {key} - {spec['error_msg']}")

        is_valid = (
            len(errors) == 0
            if self.level == ConfigValidationLevel.STRICT
            else len(missing_configs) == 0
        )

        return ConfigValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            missing_configs=missing_configs,
            invalid_configs=invalid_configs,
        )

    def _get_config_value(self, key: str) -> Optional[str]:
        """Get configuration value by key"""
        # Map config keys to SystemConfig attributes
        key_map = {
            "OPENAI_API_KEY": lambda: self.config.model.openai_api_key,
            "OPENAI_MODEL": lambda: self.config.model.openai_model,
            "VECTOR_DB_PATH": lambda: self.config.vector_db.storage_path,
        }

        if key in key_map:
            return key_map[key]()
        return os.environ.get(key)


# ========================== GLOBAL CONFIG INSTANCE ==========================

# Create global configuration instance
config = SystemConfig()


# Backward compatibility functions
def load_config() -> Dict[str, Any]:
    """Load configuration values as a dictionary for compatibility"""
    return config.to_dict()


def get_openai_model() -> str:
    return config.model.openai_model


def get_non_embedding_model() -> str:
    return config.model.openai_model


def get_embedding_model() -> str:
    return config.model.embedding_model


def get_openai_api_base() -> Optional[str]:
    return config.model.openai_api_base


def get_embedding_api_base() -> Optional[str]:
    return config.model.embedding_api_base


def get_fallback_models() -> List[str]:
    return config.model.fallback_models


def get_accounts_db_path() -> str:
    return config.database.get_accounts_db_path()


def get_model_monitoring_db_path() -> str:
    return config.database.get_model_monitoring_db_path()


def get_prompt_optimization_db_path() -> str:
    return config.database.get_prompt_optimization_db_path()


def get_cache_db_path() -> str:
    return config.database.get_cache_db_path()


# ========================== EXPORTS ==========================

__all__ = [
    "config",
    "SystemConfig",
    "DatabaseConfig",
    "ModelConfig",
    "DataFeedConfig",
    "VectorDBConfig",
    "OptimizationConfig",
    "ConfigValidator",
    "ConfigValidationResult",
    "ConfigValidationLevel",
    "Environment",
    "load_config",
    "get_openai_model",
    "get_non_embedding_model",
    "get_embedding_model",
    "get_openai_api_base",
    "get_embedding_api_base",
    "get_fallback_models",
    "get_accounts_db_path",
    "get_model_monitoring_db_path",
    "get_prompt_optimization_db_path",
    "get_cache_db_path",
]
