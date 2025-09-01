"""
⚙️ ORACLE-X Configuration Manager

Consolidated configuration management for the ORACLE-X platform.
Handles environment variables, configuration files, and defaults in a unified way.

Features:
- Hierarchical configuration loading (env vars → config files → defaults)
- Type-safe configuration access
- Configuration validation
- Environment-specific configurations (dev, test, prod)
- Database path management
- API key and model configurations
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from common_utils import load_env_file, load_json_config, get_project_root, logger

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    accounts_db: str = "data/databases/accounts.db"
    model_monitoring_db: str = "data/databases/model_monitoring.db" 
    prompt_optimization_db: str = "data/databases/prompt_optimization.db"
    cache_db: str = "data/databases/cache.db"
    
    def get_full_path(self, db_name: str) -> Path:
        """Get full path for database file"""
        path_attr = f"{db_name}_db"
        if hasattr(self, path_attr):
            relative_path = getattr(self, path_attr)
            return get_project_root() / relative_path
        raise ValueError(f"Unknown database: {db_name}")

@dataclass
class ModelConfig:
    """Model configuration settings"""
    openai_model: str = "gpt-5-mini"
    embedding_model: str = "qwen3-embedding"
    fallback_models: List[str] = field(default_factory=lambda: ["gpt-4", "gpt-3.5-turbo"])
    openai_api_base: Optional[str] = None
    embedding_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None

@dataclass
class DataFeedConfig:
    """Data feed configuration settings"""
    # Primary API Keys
    twelve_data_api_key: Optional[str] = None
    financial_modeling_prep_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    alphavantage_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
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

@dataclass
class OptimizationConfig:
    """Optimization system configuration"""
    enabled: bool = True
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    max_generations: int = 50
    performance_window_days: int = 7

@dataclass
class SystemConfig:
    """Complete system configuration"""
    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = False
    log_level: str = "INFO"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data_feeds: DataFeedConfig = field(default_factory=DataFeedConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

class ConfigurationManager:
    """Centralized configuration manager"""
    
    def __init__(self):
        self._config: Optional[SystemConfig] = None
        self._load_order = [
            self._load_from_environment,
            self._load_from_config_files,
            self._apply_defaults
        ]
    
    def get_config(self) -> SystemConfig:
        """Get the complete system configuration"""
        if self._config is None:
            self._config = self._load_configuration()
        return self._config
    
    def refresh_config(self) -> SystemConfig:
        """Refresh configuration from sources"""
        self._config = None
        return self.get_config()

    def validate_api_keys(self, config: SystemConfig) -> Dict[str, Any]:
        """Validate API keys and return validation results"""
        validation_results = {
            'valid': [],
            'missing': [],
            'warnings': [],
            'summary': {'total': 0, 'valid': 0, 'missing': 0}
        }

        # Define required API keys
        required_keys = {
            'twelve_data_api_key': 'TwelveData API Key',
            'financial_modeling_prep_api_key': 'Financial Modeling Prep API Key',
            'finnhub_api_key': 'Finnhub API Key',
            'alphavantage_api_key': 'Alpha Vantage API Key',
            'polygon_api_key': 'Polygon.io API Key',
            'twitter_bearer_token': 'Twitter Bearer Token'
        }

        # Define optional API keys
        optional_keys = {
            'iex_api_key': 'IEX API Key',
            'tradingeconomics_api_key': 'TradingEconomics API Key',
            'fred_api_key': 'FRED API Key',
            'newsapi_api_key': 'NewsAPI API Key',
            'google_news_api_key': 'Google News API Key',
            'reddit_client_id': 'Reddit Client ID',
            'reddit_client_secret': 'Reddit Client Secret'
        }

        # Validate required keys
        for field_name, display_name in required_keys.items():
            if hasattr(config.data_feeds, field_name):
                api_key = getattr(config.data_feeds, field_name)
                validation_results['summary']['total'] += 1

                if api_key and isinstance(api_key, str) and api_key.strip():
                    validation_results['valid'].append({
                        'key': field_name,
                        'name': display_name,
                        'status': 'valid'
                    })
                    validation_results['summary']['valid'] += 1
                else:
                    validation_results['missing'].append({
                        'key': field_name,
                        'name': display_name,
                        'status': 'missing',
                        'message': f'{display_name} is required but not provided'
                    })
                    validation_results['summary']['missing'] += 1

        # Validate optional keys
        for field_name, display_name in optional_keys.items():
            if hasattr(config.data_feeds, field_name):
                api_key = getattr(config.data_feeds, field_name)
                if api_key and isinstance(api_key, str) and api_key.strip():
                    validation_results['valid'].append({
                        'key': field_name,
                        'name': display_name,
                        'status': 'valid'
                    })
                    validation_results['summary']['total'] += 1
                    validation_results['summary']['valid'] += 1

        # Add warnings for missing required keys
        if validation_results['summary']['missing'] > 0:
            validation_results['warnings'].append({
                'type': 'missing_required_keys',
                'message': f"{validation_results['summary']['missing']} required API keys are missing. Some features may not work properly.",
                'severity': 'high'
            })

        # Add fallback recommendation if no keys are provided
        if validation_results['summary']['valid'] == 0:
            validation_results['warnings'].append({
                'type': 'no_api_keys',
                'message': 'No API keys configured. System will rely on fallback data sources with limited functionality.',
                'severity': 'medium'
            })

        return validation_results

    def print_validation_report(self, config: SystemConfig) -> None:
        """Print a formatted validation report"""
        results = self.validate_api_keys(config)

        print("\n" + "="*60)
        print("🔑 API KEY VALIDATION REPORT")
        print("="*60)

        # Summary
        summary = results['summary']
        print("\n📊 SUMMARY:")
        print(f"   Total API Keys: {summary['total']}")
        print(f"   Valid Keys: {summary['valid']}")
        print(f"   Missing Keys: {summary['missing']}")

        # Valid keys
        if results['valid']:
            print("\n✅ VALID API KEYS:")
            for key_info in results['valid']:
                print(f"   ✓ {key_info['name']}")

        # Missing keys
        if results['missing']:
            print("\n❌ MISSING REQUIRED API KEYS:")
            for key_info in results['missing']:
                print(f"   ✗ {key_info['name']}")

        # Warnings
        if results['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in results['warnings']:
                severity_icon = "🔴" if warning['severity'] == 'high' else "🟡"
                print(f"   {severity_icon} {warning['message']}")

        print("\n" + "="*60)
    
    def _load_configuration(self) -> SystemConfig:
        """Load configuration from all sources"""
        config_data = {}
        
        # Apply configuration in order (later sources override earlier ones)
        for loader in self._load_order:
            loader_data = loader()
            config_data.update(loader_data)
        
        return self._build_system_config(config_data)
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Environment and debug
        if os.getenv('ORACLE_ENV'):
            env_config['environment'] = os.getenv('ORACLE_ENV')
        if os.getenv('DEBUG'):
            debug_val = os.getenv('DEBUG', '')
            env_config['debug_mode'] = debug_val.lower() in ('true', '1', 'yes')
        if os.getenv('LOG_LEVEL'):
            env_config['log_level'] = os.getenv('LOG_LEVEL')
        
        # Database paths
        db_config = {}
        if os.getenv('ACCOUNTS_DB_PATH'):
            db_config['accounts_db'] = os.getenv('ACCOUNTS_DB_PATH')
        if os.getenv('MODEL_MONITORING_DB_PATH'):
            db_config['model_monitoring_db'] = os.getenv('MODEL_MONITORING_DB_PATH')
        if os.getenv('PROMPT_OPTIMIZATION_DB_PATH'):
            db_config['prompt_optimization_db'] = os.getenv('PROMPT_OPTIMIZATION_DB_PATH')
        if os.getenv('CACHE_DB_PATH'):
            db_config['cache_db'] = os.getenv('CACHE_DB_PATH')
        if db_config:
            env_config['database'] = db_config
        
        # Model configuration
        model_config = {}
        if os.getenv('OPENAI_MODEL'):
            model_config['openai_model'] = os.getenv('OPENAI_MODEL')
        if os.getenv('EMBEDDING_MODEL'):
            model_config['embedding_model'] = os.getenv('EMBEDDING_MODEL')
        if os.getenv('OPENAI_API_BASE'):
            model_config['openai_api_base'] = os.getenv('OPENAI_API_BASE')
        if os.getenv('EMBEDDING_API_BASE'):
            model_config['embedding_api_base'] = os.getenv('EMBEDDING_API_BASE')
        if os.getenv('OPENAI_API_KEY'):
            model_config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        if model_config:
            env_config['model'] = model_config
        
        # Data feed configuration
        data_config = {}

        # Primary API Keys
        if os.getenv('TWELVEDATA_API_KEY'):
            data_config['twelve_data_api_key'] = os.getenv('TWELVEDATA_API_KEY')
        if os.getenv('FINANCIALMODELINGPREP_API_KEY'):
            data_config['financial_modeling_prep_api_key'] = os.getenv('FINANCIALMODELINGPREP_API_KEY')
        if os.getenv('FINNHUB_API_KEY'):
            data_config['finnhub_api_key'] = os.getenv('FINNHUB_API_KEY')
        if os.getenv('ALPHAVANTAGE_API_KEY'):
            data_config['alphavantage_api_key'] = os.getenv('ALPHAVANTAGE_API_KEY')
        if os.getenv('POLYGON_API_KEY'):
            data_config['polygon_api_key'] = os.getenv('POLYGON_API_KEY')
        if os.getenv('TWITTER_BEARER_TOKEN'):
            data_config['twitter_bearer_token'] = os.getenv('TWITTER_BEARER_TOKEN')

        # Social Media APIs
        if os.getenv('REDDIT_CLIENT_ID'):
            data_config['reddit_client_id'] = os.getenv('REDDIT_CLIENT_ID')
        if os.getenv('REDDIT_CLIENT_SECRET'):
            data_config['reddit_client_secret'] = os.getenv('REDDIT_CLIENT_SECRET')

        # Optional API Keys
        if os.getenv('IEX_API_KEY'):
            data_config['iex_api_key'] = os.getenv('IEX_API_KEY')
        if os.getenv('TRADINGECONOMICS_API_KEY'):
            data_config['tradingeconomics_api_key'] = os.getenv('TRADINGECONOMICS_API_KEY')
        if os.getenv('FRED_API_KEY'):
            data_config['fred_api_key'] = os.getenv('FRED_API_KEY')
        if os.getenv('NEWSAPI_API_KEY'):
            data_config['newsapi_api_key'] = os.getenv('NEWSAPI_API_KEY')
        if os.getenv('GOOGLE_NEWS_API_KEY'):
            data_config['google_news_api_key'] = os.getenv('GOOGLE_NEWS_API_KEY')

        # Configuration Settings
        if os.getenv('API_REQUEST_TIMEOUT'):
            data_config['request_timeout'] = int(os.getenv('API_REQUEST_TIMEOUT', 30))
        if os.getenv('API_MAX_RETRIES'):
            data_config['max_retries'] = int(os.getenv('API_MAX_RETRIES', 3))
        if os.getenv('API_RATE_LIMIT_DELAY'):
            data_config['api_rate_limit_delay'] = float(os.getenv('API_RATE_LIMIT_DELAY', 1.0))
        if os.getenv('ENABLE_API_FALLBACK'):
            data_config['enable_api_fallback'] = os.getenv('ENABLE_API_FALLBACK', '').lower() in ('true', '1', 'yes')

        if data_config:
            env_config['data_feeds'] = data_config
        
        return env_config
    
    def _load_from_config_files(self) -> Dict[str, Any]:
        """Load configuration from config files"""
        config_data = {}
        
        # Load from multiple config files
        config_files = [
            'config/optimization.env',
            'config/rss_feeds_config.env',
            'config/optimization_config.json',
            '.env'
        ]
        
        for config_file in config_files:
            try:
                if config_file.endswith('.json'):
                    file_data = load_json_config(config_file)
                else:
                    env_data = load_env_file(config_file)
                    file_data = self._convert_env_to_config(env_data)
                
                # Merge configuration
                self._deep_merge(config_data, file_data)
                
            except Exception as e:
                logger.debug(f"Could not load config file {config_file}: {e}")
        
        return config_data
    
    def _apply_defaults(self) -> Dict[str, Any]:
        """Apply default configuration values"""
        return {
            'environment': Environment.DEVELOPMENT.value,
            'debug_mode': False,
            'log_level': 'INFO'
        }
    
    def _convert_env_to_config(self, env_data: Dict[str, str]) -> Dict[str, Any]:
        """Convert environment variables to configuration structure"""
        config = {}
        
        # Simple mappings
        simple_mappings = {
            'ORACLE_ENV': 'environment',
            'DEBUG': 'debug_mode',
            'LOG_LEVEL': 'log_level',
        }
        
        for env_key, config_key in simple_mappings.items():
            if env_key in env_data:
                value = env_data[env_key]
                if config_key == 'debug_mode':
                    value = value.lower() in ('true', '1', 'yes')
                config[config_key] = value
        
        # Database configuration
        db_keys = {
            'ACCOUNTS_DB_PATH': 'accounts_db',
            'MODEL_MONITORING_DB_PATH': 'model_monitoring_db',
            'PROMPT_OPTIMIZATION_DB_PATH': 'prompt_optimization_db',
            'CACHE_DB_PATH': 'cache_db'
        }
        
        db_config = {}
        for env_key, config_key in db_keys.items():
            if env_key in env_data:
                db_config[config_key] = env_data[env_key]
        if db_config:
            config['database'] = db_config
        
        # Model configuration
        model_keys = {
            'OPENAI_MODEL': 'openai_model',
            'EMBEDDING_MODEL': 'embedding_model',
            'OPENAI_API_BASE': 'openai_api_base',
            'EMBEDDING_API_BASE': 'embedding_api_base',
            'OPENAI_API_KEY': 'openai_api_key'
        }

        model_config = {}
        for env_key, config_key in model_keys.items():
            if env_key in env_data:
                model_config[config_key] = env_data[env_key]
        if model_config:
            config['model'] = model_config

        # Data feed configuration
        data_keys = {
            # Primary API Keys
            'TWELVEDATA_API_KEY': 'twelve_data_api_key',
            'FINANCIALMODELINGPREP_API_KEY': 'financial_modeling_prep_api_key',
            'FINNHUB_API_KEY': 'finnhub_api_key',
            'ALPHAVANTAGE_API_KEY': 'alphavantage_api_key',
            'POLYGON_API_KEY': 'polygon_api_key',
            'TWITTER_BEARER_TOKEN': 'twitter_bearer_token',

            # Social Media APIs
            'REDDIT_CLIENT_ID': 'reddit_client_id',
            'REDDIT_CLIENT_SECRET': 'reddit_client_secret',

            # Optional API Keys
            'IEX_API_KEY': 'iex_api_key',
            'TRADINGECONOMICS_API_KEY': 'tradingeconomics_api_key',
            'FRED_API_KEY': 'fred_api_key',
            'NEWSAPI_API_KEY': 'newsapi_api_key',
            'GOOGLE_NEWS_API_KEY': 'google_news_api_key',

            # Configuration Settings
            'API_REQUEST_TIMEOUT': 'request_timeout',
            'API_MAX_RETRIES': 'max_retries',
            'API_RATE_LIMIT_DELAY': 'api_rate_limit_delay',
            'ENABLE_API_FALLBACK': 'enable_api_fallback'
        }

        data_config = {}
        for env_key, config_key in data_keys.items():
            if env_key in env_data:
                value = env_data[env_key]
                # Convert string values to appropriate types
                if config_key in ['request_timeout', 'max_retries']:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        value = 30 if config_key == 'request_timeout' else 3
                elif config_key == 'api_rate_limit_delay':
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = 1.0
                elif config_key == 'enable_api_fallback':
                    value = value.lower() in ('true', '1', 'yes')
                data_config[config_key] = value

        if data_config:
            config['data_feeds'] = data_config
        
        return config
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge source into target dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _build_system_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """Build SystemConfig from merged configuration data"""
        # Handle environment
        env_str = config_data.get('environment', Environment.DEVELOPMENT.value)
        if isinstance(env_str, str):
            try:
                environment = Environment(env_str)
            except ValueError:
                environment = Environment.DEVELOPMENT
        else:
            environment = env_str
        
        # Build database config - filter known fields
        db_data = config_data.get('database', {})
        db_fields = {'accounts_db', 'model_monitoring_db', 'prompt_optimization_db', 'cache_db'}
        filtered_db_data = {k: v for k, v in db_data.items() if k in db_fields}
        database = DatabaseConfig(**filtered_db_data)
        
        # Build model config - filter known fields
        model_data = config_data.get('model', {})
        model_fields = {'openai_model', 'embedding_model', 'fallback_models', 'openai_api_base', 'embedding_api_base', 'openai_api_key'}
        filtered_model_data = {k: v for k, v in model_data.items() if k in model_fields}
        model = ModelConfig(**filtered_model_data)
        
        # Build data feeds config - filter known fields
        data_feeds_data = config_data.get('data_feeds', {})
        data_fields = {
            # Primary API Keys
            'twelve_data_api_key', 'financial_modeling_prep_api_key', 'finnhub_api_key',
            'alphavantage_api_key', 'polygon_api_key', 'twitter_bearer_token',
            # Social Media APIs
            'reddit_client_id', 'reddit_client_secret',
            # Optional API Keys
            'iex_api_key', 'tradingeconomics_api_key', 'fred_api_key',
            'newsapi_api_key', 'google_news_api_key',
            # Configuration Settings
            'cache_ttl_minutes', 'max_retries', 'request_timeout',
            'api_rate_limit_delay', 'enable_api_fallback'
        }
        filtered_data_feeds_data = {k: v for k, v in data_feeds_data.items() if k in data_fields}
        data_feeds = DataFeedConfig(**filtered_data_feeds_data)
        
        # Build optimization config - filter known fields
        opt_data = config_data.get('optimization', {})
        opt_fields = {'enabled', 'population_size', 'mutation_rate', 'crossover_rate', 'max_generations', 'performance_window_days'}
        filtered_opt_data = {k: v for k, v in opt_data.items() if k in opt_fields}
        optimization = OptimizationConfig(**filtered_opt_data)
        
        return SystemConfig(
            environment=environment,
            debug_mode=config_data.get('debug_mode', False),
            log_level=config_data.get('log_level', 'INFO'),
            database=database,
            model=model,
            data_feeds=data_feeds,
            optimization=optimization
        )

# Global configuration manager instance
_config_manager = ConfigurationManager()

# Convenience functions for backward compatibility
def get_config() -> SystemConfig:
    """Get the complete system configuration"""
    return _config_manager.get_config()

def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return get_config().database

def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return get_config().model

def get_data_feeds_config() -> DataFeedConfig:
    """Get data feeds configuration"""
    return get_config().data_feeds

def get_optimization_config() -> OptimizationConfig:
    """Get optimization configuration"""
    return get_config().optimization

# Data feed API key convenience functions
def get_twelve_data_api_key() -> Optional[str]:
    """Get TwelveData API key"""
    return get_data_feeds_config().twelve_data_api_key

def get_financial_modeling_prep_api_key() -> Optional[str]:
    """Get Financial Modeling Prep API key"""
    return get_data_feeds_config().financial_modeling_prep_api_key

def get_finnhub_api_key() -> Optional[str]:
    """Get Finnhub API key"""
    return get_data_feeds_config().finnhub_api_key

def get_alphavantage_api_key() -> Optional[str]:
    """Get Alpha Vantage API key"""
    return get_data_feeds_config().alphavantage_api_key

def get_polygon_api_key() -> Optional[str]:
    """Get Polygon.io API key"""
    return get_data_feeds_config().polygon_api_key

def get_twitter_bearer_token() -> Optional[str]:
    """Get Twitter Bearer Token"""
    return get_data_feeds_config().twitter_bearer_token

def get_reddit_client_id() -> Optional[str]:
    """Get Reddit Client ID"""
    return get_data_feeds_config().reddit_client_id

def get_reddit_client_secret() -> Optional[str]:
    """Get Reddit Client Secret"""
    return get_data_feeds_config().reddit_client_secret

def validate_api_key_configuration() -> Dict[str, Any]:
    """Validate the current API key configuration"""
    return _config_manager.validate_api_keys(get_config())

def print_api_key_validation_report() -> None:
    """Print a formatted API key validation report"""
    _config_manager.print_validation_report(get_config())

# Backward compatibility functions
def get_cache_db_path() -> str:
    """Get cache database path (backward compatibility)"""
    return get_database_config().cache_db

def get_accounts_db_path() -> str:
    """Get accounts database path (backward compatibility)"""
    return get_database_config().accounts_db

def get_model_monitoring_db_path() -> str:
    """Get model monitoring database path"""
    return get_database_config().model_monitoring_db

def get_prompt_optimization_db_path() -> str:
    """Get prompt optimization database path"""
    return get_database_config().prompt_optimization_db

# Model configuration functions (env_config.py compatibility)
def get_openai_model() -> str:
    """Get OpenAI model name"""
    return get_model_config().openai_model

def get_non_embedding_model() -> str:
    """Get non-embedding model (backward compatibility)"""
    return get_openai_model()

def get_embedding_model() -> str:
    """Get embedding model name"""
    return get_model_config().embedding_model

def get_openai_api_base() -> Optional[str]:
    """Get OpenAI API base URL"""
    return get_model_config().openai_api_base

def get_embedding_api_base() -> Optional[str]:
    """Get embedding API base URL"""
    return get_model_config().embedding_api_base

def get_fallback_models() -> List[str]:
    """Get fallback models list"""
    return get_model_config().fallback_models

def resolve_model(client, preferred_model: str, test: bool = False) -> str:
    """Resolve model with fallback logic (env_config.py compatibility)"""
    # For now, return the preferred model
    # This can be enhanced with actual model validation logic if needed
    return preferred_model or get_openai_model()

def load_config():
    """Load configuration (backward compatibility)"""
    return get_config()

# Legacy support for existing imports
OPENAI_MODEL = get_model_config().openai_model
EMBEDDING_MODEL = get_model_config().embedding_model
CACHE_DB_PATH = get_cache_db_path()
ACCOUNTS_DB_PATH = get_accounts_db_path()
MODEL_MONITORING_DB_PATH = get_database_config().model_monitoring_db
PROMPT_OPTIMIZATION_DB_PATH = get_database_config().prompt_optimization_db

# Export configuration classes
__all__ = [
    # Main configuration functions
    'get_config', 'get_database_config', 'get_model_config',
    'get_data_feeds_config', 'get_optimization_config',
    'validate_api_key_configuration', 'print_api_key_validation_report',

    # Database path functions
    'get_cache_db_path', 'get_accounts_db_path', 'get_model_monitoring_db_path', 'get_prompt_optimization_db_path',

    # Model configuration functions
    'get_openai_model', 'get_non_embedding_model', 'get_embedding_model',
    'get_openai_api_base', 'get_embedding_api_base', 'get_fallback_models',
    'resolve_model', 'load_config',

    # Data feed API key functions
    'get_twelve_data_api_key', 'get_financial_modeling_prep_api_key', 'get_finnhub_api_key',
    'get_alphavantage_api_key', 'get_polygon_api_key', 'get_twitter_bearer_token',
    'get_reddit_client_id', 'get_reddit_client_secret',

    # Configuration classes
    'SystemConfig', 'DatabaseConfig', 'ModelConfig', 'DataFeedConfig', 'OptimizationConfig',
    'Environment', 'ConfigurationManager',

    # Legacy exports
    'OPENAI_MODEL', 'EMBEDDING_MODEL', 'CACHE_DB_PATH', 'ACCOUNTS_DB_PATH',
    'MODEL_MONITORING_DB_PATH', 'PROMPT_OPTIMIZATION_DB_PATH'
]
