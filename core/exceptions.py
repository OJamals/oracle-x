"""
Custom exception hierarchy for Oracle-X project.
All custom exceptions inherit from OracleException.
"""


class OracleException(Exception):
    """Base exception class for all Oracle-X specific errors."""

    pass


class DataFeedError(OracleException):
    """Raised when data feed operations fail."""

    pass


class CacheError(OracleException):
    """Raised when cache operations fail."""

    pass


class APIError(OracleException):
    """Raised for API communication errors (requests, timeouts, etc.)."""

    pass


class ValidationError(OracleException):
    """Raised for data validation failures."""

    pass


class ConfigError(OracleException):
    """Raised for configuration loading or validation errors."""

    pass


class MLPredictionError(OracleException):
    """Raised for ML model prediction errors."""

    pass
