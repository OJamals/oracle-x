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
import json
from typing import Dict, Any, Optional, Union, List
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
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    fallback_models: List[str] = field(default_factory=lambda: ["gpt-4", "gpt-3.5-turbo"])
    openai_api_base: Optional[str] = None
    embedding_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None

@dataclass
class DataFeedConfig:
    """Data feed configuration settings"""
    twelve_data_api_key: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    cache_ttl_minutes: int = 5
    max_retries: int = 3
    request_timeout: int = 30

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
        if os.getenv('TWELVEDATA_API_KEY'):
            data_config['twelve_data_api_key'] = os.getenv('TWELVEDATA_API_KEY')
        if os.getenv('REDDIT_CLIENT_ID'):
            data_config['reddit_client_id'] = os.getenv('REDDIT_CLIENT_ID')
        if os.getenv('REDDIT_CLIENT_SECRET'):
            data_config['reddit_client_secret'] = os.getenv('REDDIT_CLIENT_SECRET')
        if os.getenv('TWITTER_BEARER_TOKEN'):
            data_config['twitter_bearer_token'] = os.getenv('TWITTER_BEARER_TOKEN')
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
        data_fields = {'twelve_data_api_key', 'reddit_client_id', 'reddit_client_secret', 'twitter_bearer_token', 'cache_ttl_minutes', 'max_retries', 'request_timeout'}
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
    'get_config', 'get_database_config', 'get_model_config', 
    'get_data_feeds_config', 'get_optimization_config',
    'get_cache_db_path', 'get_accounts_db_path', 'get_model_monitoring_db_path', 'get_prompt_optimization_db_path',
    'get_openai_model', 'get_non_embedding_model', 'get_embedding_model',
    'get_openai_api_base', 'get_embedding_api_base', 'get_fallback_models',
    'resolve_model', 'load_config',
    'SystemConfig', 'DatabaseConfig', 'ModelConfig', 'DataFeedConfig', 'OptimizationConfig',
    'Environment', 'ConfigurationManager',
    # Legacy exports
    'OPENAI_MODEL', 'EMBEDDING_MODEL', 'CACHE_DB_PATH', 'ACCOUNTS_DB_PATH',
    'MODEL_MONITORING_DB_PATH', 'PROMPT_OPTIMIZATION_DB_PATH'
]
