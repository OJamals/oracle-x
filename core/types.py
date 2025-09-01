"""
Core type definitions and validation system for Oracle-X financial trading system.
Provides standardized data structures with comprehensive validation for market data,
options contracts, and data sources.

Enhanced with additional validation, quality metrics, and extended metadata for Phase 2.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import re
from pydantic import BaseModel, Field, validator



class DataSource(Enum):
    """Enum representing different data sources with priority and rate limiting metadata."""
    YFINANCE = auto()
    TWELVE_DATA = auto()
    FMP = auto()
    FINNHUB = auto()
    ALPHA_VANTAGE = auto()
    
    @property
    def priority(self) -> int:
        """Get the priority of this data source (lower number = higher priority)."""
        priorities = {
            DataSource.YFINANCE: 1,
            DataSource.TWELVE_DATA: 2,
            DataSource.FMP: 3,
            DataSource.FINNHUB: 4,
            DataSource.ALPHA_VANTAGE: 5
        }
        return priorities[self]
    
    @property
    def requests_per_second(self) -> Optional[float]:
        """Get the rate limit for this data source in requests per second."""
        rate_limits = {
            DataSource.YFINANCE: None,  # Unlimited
            DataSource.TWELVE_DATA: 5.0,
            DataSource.FMP: 0.0029,  # 250/day
            DataSource.FINNHUB: 1.0,
            DataSource.ALPHA_VANTAGE: 0.083  # 5/min
        }
        return rate_limits[self]


class OptionType(Enum):
    """Enum representing option contract types."""
    CALL = "CALL"
    PUT = "PUT"


class OptionStyle(Enum):
    """Enum representing option exercise styles."""
    AMERICAN = "AMERICAN"
    EUROPEAN = "EUROPEAN"


class MarketData(BaseModel):
    """
    Standardized market data structure with comprehensive validation.
    """
    symbol: str = Field(..., description="Stock symbol in uppercase format")
    timestamp: datetime = Field(..., description="UTC timestamp of the data")
    open: Decimal = Field(..., gt=0, decimal_places=4, description="Open price")
    high: Decimal = Field(..., gt=0, decimal_places=4, description="High price")
    low: Decimal = Field(..., gt=0, decimal_places=4, description="Low price")
    close: Decimal = Field(..., gt=0, decimal_places=4, description="Close price")
    volume: int = Field(..., ge=0, description="Trading volume")
    source: DataSource = Field(..., description="Data source provider")
    cache_ttl: int = Field(default=300, ge=0, description="Cache TTL in seconds")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality score (0.0-1.0)")
    
    @validator('symbol')
    def validate_symbol_format(cls, v):
        """Validate that symbol is in correct format (uppercase, no special chars)."""
        if not re.match(r'^[A-Z]+$', v):
            raise ValueError(f"Invalid symbol format: {v}. Must be uppercase letters only.")
        return v
    
    @validator('timestamp')
    def validate_timestamp_utc(cls, v):
        """Ensure timestamp is in UTC timezone."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must have timezone information")
        if v.tzinfo.utcoffset(v) != v.utcoffset():
            raise ValueError("Timestamp must be in UTC timezone")
        return v
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class OptionContract(BaseModel):
    """
    Standardized option contract structure with financial validation.
    """
    symbol: str = Field(..., description="Option symbol")
    strike: Decimal = Field(..., gt=0, decimal_places=2, description="Strike price")
    expiry: datetime = Field(..., description="Expiration date")
    option_type: OptionType = Field(..., description="Call or Put option")
    style: OptionStyle = Field(default=OptionStyle.AMERICAN, description="Exercise style")
    bid: Optional[Decimal] = Field(None, ge=0, decimal_places=4, description="Bid price")
    ask: Optional[Decimal] = Field(None, ge=0, decimal_places=4, description="Ask price")
    last: Optional[Decimal] = Field(None, ge=0, decimal_places=4, description="Last trade price")
    volume: Optional[int] = Field(None, ge=0, description="Trading volume")
    open_interest: Optional[int] = Field(None, ge=0, description="Open interest")
    implied_volatility: Optional[Decimal] = Field(None, ge=0, le=5, decimal_places=4, description="Implied volatility (0-5.0)")
    underlying_price: Optional[Decimal] = Field(None, gt=0, decimal_places=4, description="Underlying asset price")
    
    @validator('symbol')
    def validate_option_symbol(cls, v):
        """Validate option symbol format."""
        if not re.match(r'^[A-Z]+\d{6}[CP]\d{8}$', v):
            raise ValueError(f"Invalid option symbol format: {v}")
        return v
    
    @validator('expiry')
    def validate_expiry_future(cls, v):
        """Ensure expiry date is reasonable (not in the distant past)."""
        # Allow historical data for backtesting, but not dates before 2000
        if v.year < 2000:
            raise ValueError("Expiry date is too far in the past")
        return v
    
    @validator('implied_volatility')
    def validate_implied_volatility_range(cls, v):
        """Validate implied volatility is within reasonable bounds."""
        if v is not None and (v < 0 or v > 5.0):
            raise ValueError("Implied volatility must be between 0 and 5.0")
        return v
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


def validate_market_data(data: Dict[str, Any]) -> MarketData:
    """
    Comprehensive validation function for market data.
    
    Args:
        data: Dictionary containing market data fields
        
    Returns:
        Validated MarketData object
        
    Raises:
        ValueError: If data fails validation
    """
    try:
        return MarketData(**data)
    except Exception as e:
        raise ValueError(f"Market data validation failed: {str(e)}")


def validate_option_contract(data: Dict[str, Any]) -> OptionContract:
    """
    Comprehensive validation function for option contracts.
    
    Args:
        data: Dictionary containing option contract fields
        
    Returns:
        Validated OptionContract object
        
    Raises:
        ValueError: If data fails validation
    """
    try:
        return OptionContract(**data)
    except Exception as e:
        raise ValueError(f"Option contract validation failed: {str(e)}")


def calculate_data_quality(data: Any) -> float:
    """
    Calculate data quality score for any financial data object.
    
    Args:
        data: MarketData, OptionContract, or other financial data object
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    if isinstance(data, MarketData):
        return _calculate_market_data_quality(data)
    elif isinstance(data, OptionContract):
        return _calculate_option_quality(data)
    else:
        # Default quality score for unknown types
        return 0.5


def _calculate_market_data_quality(data: MarketData) -> float:
    """Calculate quality score for MarketData."""
    score = 1.0
    
    # Check for missing critical fields
    if data.volume == 0:
        score *= 0.8
    
    # Check price consistency
    if data.low > data.high:
        score *= 0.5
    if data.open > data.high or data.open < data.low:
        score *= 0.7
    if data.close > data.high or data.close < data.low:
        score *= 0.7
    
    # Check for stale data (more than 1 hour old)
    age_hours = (datetime.now(data.timestamp.tzinfo) - data.timestamp).total_seconds() / 3600
    if age_hours > 1:
        score *= max(0.8, 1.0 - (age_hours - 1) * 0.1)
    
    return max(0.0, min(1.0, score))


def _calculate_option_quality(data: OptionContract) -> float:
    """Calculate quality score for OptionContract."""
    score = 1.0
    
    # Check for missing pricing data
    missing_fields = 0
    total_fields = 3  # bid, ask, last
    
    if data.bid is None:
        missing_fields += 1
    if data.ask is None:
        missing_fields += 1
    if data.last is None:
        missing_fields += 1
    
    if missing_fields > 0:
        score *= (total_fields - missing_fields) / total_fields
    
    # Check bid-ask spread sanity
    if data.bid is not None and data.ask is not None:
        spread = data.ask - data.bid
        if spread <= 0:
            score *= 0.5
        elif data.underlying_price is not None and spread > data.underlying_price * Decimal('0.1'):
            score *= 0.7
    
    return max(0.0, min(1.0, score))


# Utility functions for data conversion
def market_data_to_dict(data: MarketData) -> Dict[str, Any]:
    """Convert MarketData to dictionary."""
    return data.dict()


def option_contract_to_dict(data: OptionContract) -> Dict[str, Any]:
    """Convert OptionContract to dictionary."""
    return data.dict()


def dict_to_market_data(data: Dict[str, Any]) -> MarketData:
    """Convert dictionary to MarketData with validation."""
    return validate_market_data(data)


def dict_to_option_contract(data: Dict[str, Any]) -> OptionContract:
    """Convert dictionary to OptionContract with validation."""
    return validate_option_contract(data)


class MarketDataExtended(MarketData):
    """
    Extended MarketData with additional validation and quality metrics.
    Enhanced with data quality assessment, source reliability, and metadata.
    """
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Enhanced data quality score (0.0-1.0)")
    source_reliability: float = Field(default=0.8, ge=0.0, le=1.0, description="Data source reliability rating (0.0-1.0)")
    freshness_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Time-based freshness metric (0.0-1.0)")
    completeness_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Field completeness assessment (0.0-1.0)")
    validation_errors: List[str] = Field(default_factory=list, description="List of validation issues found")
    
    @validator('data_quality_score')
    def validate_quality_score(cls, v):
        """Validate quality score is within bounds."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Data quality score must be between 0.0 and 1.0")
        return v
    
    @validator('source_reliability')
    def validate_source_reliability(cls, v):
        """Validate source reliability is within bounds."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Source reliability must be between 0.0 and 1.0")
        return v


class OptionContractEnhanced(OptionContract):
    """
    Enhanced OptionContract with Greeks, risk metrics, and additional validation.
    """
    delta: Optional[Decimal] = Field(None, ge=-1.0, le=1.0, decimal_places=4, description="Delta Greek value")
    gamma: Optional[Decimal] = Field(None, ge=0.0, le=1.0, decimal_places=4, description="Gamma Greek value")
    theta: Optional[Decimal] = Field(None, le=0.0, decimal_places=4, description="Theta Greek value")
    vega: Optional[Decimal] = Field(None, ge=0.0, decimal_places=4, description="Vega Greek value")
    rho: Optional[Decimal] = Field(None, decimal_places=4, description="Rho Greek value")
    iv_surface_position: Optional[Tuple[float, float]] = Field(None, description="Position in IV surface matrix (moneyness, days_to_expiry)")
    liquidity_score: float = Field(default=50.0, ge=0.0, le=100.0, description="Combined volume/OI liquidity score (0-100)")
    spread_ratio: Optional[float] = Field(None, ge=0.0, description="Bid-ask spread as percentage of underlying")
    time_decay_risk: float = Field(default=0.5, ge=0.0, le=1.0, description="Time decay risk assessment (0.0-1.0)")
    
    @validator('delta')
    def validate_delta_range(cls, v):
        """Validate delta is within reasonable bounds."""
        if v is not None and (v < -1.0 or v > 1.0):
            raise ValueError("Delta must be between -1.0 and 1.0")
        return v
    
    @validator('gamma')
    def validate_gamma_range(cls, v):
        """Validate gamma is within reasonable bounds."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Gamma must be between 0.0 and 1.0")
        return v
    
    @validator('theta')
    def validate_theta_range(cls, v):
        """Validate theta is non-positive (time decay)."""
        if v is not None and v > 0.0:
            raise ValueError("Theta must be non-positive (time decay)")
        return v
    
    @validator('vega')
    def validate_vega_range(cls, v):
        """Validate vega is non-negative."""
        if v is not None and v < 0.0:
            raise ValueError("Vega must be non-negative")
        return v
    
    @validator('liquidity_score')
    def validate_liquidity_score(cls, v):
        """Validate liquidity score is within bounds."""
        if v < 0.0 or v > 100.0:
            raise ValueError("Liquidity score must be between 0.0 and 100.0")
        return v
    
    @validator('time_decay_risk')
    def validate_time_decay_risk(cls, v):
        """Validate time decay risk is within bounds."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Time decay risk must be between 0.0 and 1.0")
        return v


@dataclass
class DataSourceMetrics:
    """
    Comprehensive data source performance tracking and metrics.
    """
    success_rate: float = 0.0  # API call success percentage (0.0-1.0)
    avg_response_time: float = 0.0  # Average response time in milliseconds
    reliability_score: float = 0.5  # Overall reliability score (0.0-1.0)
    rate_limit_utilization: float = 0.0  # Current rate limit usage (0.0-1.0)
    error_types: Dict[str, int] = field(default_factory=dict)  # Categorized error statistics
    
    def update_success(self, response_time: float):
        """Update metrics for a successful API call."""
        self.success_rate = (self.success_rate * 0.9) + 0.1  # Exponential moving average
        self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
        self.reliability_score = min(1.0, self.reliability_score + 0.01)
    
    def update_failure(self, error_type: str):
        """Update metrics for a failed API call."""
        self.success_rate = self.success_rate * 0.9  # Exponential moving average
        self.reliability_score = max(0.0, self.reliability_score - 0.05)
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    def get_error_summary(self) -> str:
        """Get summary of error types."""
        if not self.error_types:
            return "No errors recorded"
        return ", ".join([f"{k}: {v}" for k, v in self.error_types.items()])
