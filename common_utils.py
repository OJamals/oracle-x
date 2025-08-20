"""
🔧 ORACLE-X Common Utilities

Consolidated utility functions and patterns used across the ORACLE-X codebase.
Eliminates code duplication and provides standardized functionality.

Common Patterns Consolidated:
- Configuration management
- Logging setup  
- Path resolution
- Database connection handling
- Error handling patterns
- Performance monitoring
- CLI formatting utilities
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps
import sqlite3
from contextlib import contextmanager

# ========================== PATH AND PROJECT MANAGEMENT ==========================

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent

def setup_project_path():
    """Add project root to Python path if not already present"""
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def resolve_project_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root"""
    return get_project_root() / relative_path

# ========================== CONFIGURATION MANAGEMENT ==========================

def load_env_file(env_file: str) -> Dict[str, str]:
    """Load environment variables from a file"""
    env_vars = {}
    env_path = resolve_project_path(env_file)
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip().strip('"\'')
    
    return env_vars

def get_config_value(key: str, default: Any = None, config_file: Optional[str] = None) -> Any:
    """Get configuration value from environment or config file"""
    # First check environment variables
    value = os.getenv(key)
    if value is not None:
        return value
    
    # If config file specified, check it
    if config_file:
        env_vars = load_env_file(config_file)
        if key in env_vars:
            return env_vars[key]
    
    # Try common config files
    for config_file in ['config/.env', '.env', 'config/optimization.env']:
        env_vars = load_env_file(config_file)
        if key in env_vars:
            return env_vars[key]
    
    return default

def load_json_config(config_file: str) -> Dict[str, Any]:
    """Load JSON configuration file"""
    config_path = resolve_project_path(config_file)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

# ========================== LOGGING SETUP ==========================

def setup_logging(
    name: str = "oracle-x",
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    suppress_warnings: bool = True
) -> logging.Logger:
    """Setup standardized logging configuration"""
    
    if suppress_warnings:
        warnings.filterwarnings('ignore')
    
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger(name)
    return logger

# ========================== DATABASE UTILITIES ==========================

@contextmanager
def get_database_connection(db_path: str):
    """Get database connection with proper resource management"""
    full_path = resolve_project_path(db_path)
    
    # Ensure directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = None
    try:
        conn = sqlite3.connect(full_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        yield conn
    finally:
        if conn:
            conn.close()

def execute_sql_query(db_path: str, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """Execute SQL query and return results as list of dictionaries"""
    with get_database_connection(db_path) as conn:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        # Convert rows to dictionaries
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

def execute_sql_command(db_path: str, command: str, params: Optional[tuple] = None) -> int:
    """Execute SQL command (INSERT, UPDATE, DELETE) and return affected rows"""
    with get_database_connection(db_path) as conn:
        cursor = conn.cursor()
        if params:
            cursor.execute(command, params)
        else:
            cursor.execute(command)
        conn.commit()
        return cursor.rowcount

# ========================== ERROR HANDLING PATTERNS ==========================

def safe_execute(func: Callable, *args, default_return=None, log_errors: bool = True, **kwargs):
    """Safely execute a function with standardized error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in {func.__name__}: {e}")
        return default_return

def retry_on_failure(retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying functions on failure with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise last_exception
            
            return None
        return wrapper
    return decorator

# ========================== PERFORMANCE MONITORING ==========================

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation", log_result: bool = True):
        self.name = name
        self.log_result = log_result
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.start_time is not None:
            self.duration = self.end_time - self.start_time
        else:
            self.duration = 0.0
        
        if self.log_result:
            logger = logging.getLogger(__name__)
            logger.info(f"{self.name} completed in {self.duration:.3f} seconds")

def time_function(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with PerformanceTimer(f"Function {func.__name__}"):
            return func(*args, **kwargs)
    return wrapper

# ========================== CLI FORMATTING UTILITIES ==========================

class CLIFormatter:
    """Standardized CLI output formatting"""
    
    @staticmethod
    def header(title: str, width: int = 70):
        """Print formatted header"""
        return f"\n{'='*width}\n  🚀 ORACLE-X: {title.upper()}\n{'='*width}"
    
    @staticmethod
    def section(title: str, width: int = 50):
        """Print formatted section"""
        return f"\n🔹 {title}\n{'-' * width}"
    
    @staticmethod
    def success(message: str):
        """Format success message"""
        return f"✅ SUCCESS: {message}"
    
    @staticmethod
    def error(message: str):
        """Format error message"""
        return f"❌ ERROR: {message}"
    
    @staticmethod
    def info(message: str):
        """Format info message"""
        return f"ℹ️  INFO: {message}"
    
    @staticmethod
    def warning(message: str):
        """Format warning message"""
        return f"⚠️  WARNING: {message}"
    
    @staticmethod
    def table_row(columns: List[str], widths: List[int], separator: str = " | "):
        """Format table row with specified column widths"""
        formatted_cols = []
        for col, width in zip(columns, widths):
            formatted_cols.append(str(col).ljust(width)[:width])
        return separator.join(formatted_cols)
    
    @staticmethod
    def progress_bar(current: int, total: int, width: int = 50, prefix: str = "Progress"):
        """Generate progress bar"""
        if total == 0:
            percentage = 100
        else:
            percentage = (current / total) * 100
        
        filled_width = int(width * current // total) if total > 0 else width
        bar = "█" * filled_width + "░" * (width - filled_width)
        return f"{prefix}: [{bar}] {percentage:.1f}% ({current}/{total})"

# ========================== DATA VALIDATION UTILITIES ==========================

def validate_schema(data: Dict[str, Any], required_fields: List[str], optional_fields: Optional[List[str]] = None) -> tuple[bool, List[str]]:
    """Validate data against a simple schema"""
    errors = []
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif data[field] is None:
            errors.append(f"Required field '{field}' cannot be None")
    
    # Check for unexpected fields
    if optional_fields is not None:
        allowed_fields = set(required_fields + optional_fields)
        for field in data.keys():
            if field not in allowed_fields:
                errors.append(f"Unexpected field: {field}")
    
    return len(errors) == 0, errors

def clean_json_for_serialization(obj: Any) -> Any:
    """Clean object for JSON serialization"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: clean_json_for_serialization(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_json_for_serialization(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: clean_json_for_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, timedelta):
        return str(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj

# ========================== COMMON PATTERNS ==========================

def ensure_directory_exists(path: Union[str, Path]):
    """Ensure directory exists, creating it if necessary"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)

def get_file_age_days(file_path: Union[str, Path]) -> Optional[float]:
    """Get file age in days, return None if file doesn't exist"""
    path_obj = Path(file_path)
    if not path_obj.exists():
        return None
    
    mod_time = datetime.fromtimestamp(path_obj.stat().st_mtime)
    age = datetime.now() - mod_time
    return age.total_seconds() / 86400  # Convert to days

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to max length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# ========================== MODULE INITIALIZATION ==========================

# Setup project path on import
setup_project_path()

# Create default logger
logger = setup_logging()

# Export commonly used utilities
__all__ = [
    'get_project_root', 'setup_project_path', 'resolve_project_path',
    'load_env_file', 'get_config_value', 'load_json_config',
    'setup_logging', 'get_database_connection', 'execute_sql_query', 'execute_sql_command',
    'safe_execute', 'retry_on_failure', 'PerformanceTimer', 'time_function',
    'CLIFormatter', 'validate_schema', 'clean_json_for_serialization',
    'ensure_directory_exists', 'get_file_age_days', 'truncate_string',
    'logger'
]
