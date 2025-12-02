"""
FinanceToolkit and FinanceDatabase Integration for Consolidated Data Feed
Adds comprehensive financial analysis capabilities using FinanceToolkit
"""

import os
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
import pandas as pd
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from financetoolkit import Toolkit

try:
    from financetoolkit import Toolkit
    FINANCE_TOOLKIT_AVAILABLE = True
except ImportError:
    FINANCE_TOOLKIT_AVAILABLE = False
    logging.warning("FinanceToolkit not available. Install with: pip install financetoolkit")

try:
    import financedatabase as fd
    FINANCE_DATABASE_AVAILABLE = True
except ImportError:
    FINANCE_DATABASE_AVAILABLE = False
    logging.warning("FinanceDatabase not available. Install with: pip install financedatabase")

try:
    from core.config import config as _core_config
except Exception:
    _core_config = None

logger = logging.getLogger(__name__)

# ============================================================================
# Enhanced Data Models for Financial Analysis
# ============================================================================

@dataclass
class FinancialRatios:
    """Comprehensive financial ratios from FinanceToolkit"""
    symbol: str
    period: str
    
    # Profitability Ratios
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_profit_margin: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_invested_capital: Optional[float] = None
    
    # Liquidity Ratios
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    working_capital: Optional[float] = None
    
    # Solvency Ratios
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    equity_multiplier: Optional[float] = None
    interest_coverage_ratio: Optional[float] = None
    
    # Efficiency Ratios
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    days_sales_outstanding: Optional[float] = None
    
    # Valuation Ratios
    price_to_earnings: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    enterprise_value_to_ebitda: Optional[float] = None
    
    timestamp: Optional[datetime] = None
    source: str = "financetoolkit"

@dataclass
class SecurityInfo:
    """Enhanced security information from FinanceDatabase"""
    symbol: str
    name: str
    asset_class: str  # equity, etf, fund, index, currency, crypto
    
    # Basic Info
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    
    # Financial Metrics
    market_cap: Optional[str] = None
    market_cap_group: Optional[str] = None
    
    # ETF/Fund specific
    category_group: Optional[str] = None
    category: Optional[str] = None
    family: Optional[str] = None
    expense_ratio: Optional[float] = None
    
    # Additional metadata
    figi: Optional[str] = None
    composite_figi: Optional[str] = None
    shareclass_figi: Optional[str] = None
    
    source: str = "financedatabase"

# ============================================================================
# FinanceToolkit Adapter
# ============================================================================

class FinanceToolkitAdapter:
    """Adapter for FinanceToolkit integration"""
    
    def __init__(self, cache, rate_limiter):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.available = FINANCE_TOOLKIT_AVAILABLE
        
        if not self.available:
            logger.warning("FinanceToolkit not available")
            return
        
        # Check for API key (FMP is recommended for full functionality)
        config_data = getattr(_core_config, 'data_feeds', None)
        self.api_key = (
            os.getenv('FINANCIALMODELINGPREP_API_KEY')
            or os.getenv('FMP_API_KEY')
            or (getattr(config_data, 'financial_modeling_prep_api_key', None) if config_data else None)
        )
        if not self.api_key:
            logger.info("No FMP API key found. FinanceToolkit will use free data sources with limitations")
    
    def create_toolkit(self, symbols: Union[str, List[str]], 
                      start_date: str = "2023-01-01", 
                      end_date: Optional[str] = None,
                      quarterly: bool = True):
        """Create a Toolkit instance for financial analysis"""
        if not self.available or not FINANCE_TOOLKIT_AVAILABLE:
            return None
        
        try:
            from financetoolkit import Toolkit
            
            if isinstance(symbols, str):
                symbols = [symbols]
            
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Initialize toolkit
            toolkit_params = {
                'tickers': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'quarterly': quarterly
            }
            
            # Add API key if available
            if self.api_key:
                toolkit_params['api_key'] = self.api_key
            
            toolkit = Toolkit(**toolkit_params)
            return toolkit
            
        except Exception as e:
            logger.error(f"Failed to create toolkit: {e}")
            return None
    
    def get_financial_ratios(self, symbols: Union[str, List[str]], 
                           period: str = "annual") -> Dict[str, FinancialRatios]:
        """Get comprehensive financial ratios for symbols"""
        if not self.available:
            return {}
        
        try:
            toolkit = self.create_toolkit(symbols)
            if not toolkit:
                return {}
            
            results = {}
            
            # Get different types of ratios
            try:
                profitability = toolkit.ratios.collect_profitability_ratios()
                liquidity = toolkit.ratios.collect_liquidity_ratios()
                solvency = toolkit.ratios.collect_solvency_ratios()
                efficiency = toolkit.ratios.collect_efficiency_ratios()
                valuation = toolkit.ratios.collect_valuation_ratios()
                
                # Process each symbol
                if isinstance(symbols, str):
                    symbols = [symbols]
                
                for symbol in symbols:
                    ratios = FinancialRatios(
                        symbol=symbol,
                        period=period,
                        timestamp=datetime.now()
                    )
                    
                    # Extract profitability ratios
                    if profitability is not None and not profitability.empty:
                        for ratio_name, ratio_value in profitability.items():
                            if symbol.upper() in str(ratio_name).upper():
                                if 'Gross Margin' in str(ratio_name):
                                    ratios.gross_margin = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Operating Margin' in str(ratio_name):
                                    ratios.operating_margin = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Net Profit Margin' in str(ratio_name):
                                    ratios.net_profit_margin = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Return on Assets' in str(ratio_name):
                                    ratios.return_on_assets = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Return on Equity' in str(ratio_name):
                                    ratios.return_on_equity = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Return on Invested Capital' in str(ratio_name):
                                    ratios.return_on_invested_capital = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                    
                    # Extract liquidity ratios
                    if liquidity is not None and not liquidity.empty:
                        for ratio_name, ratio_value in liquidity.items():
                            if symbol.upper() in str(ratio_name).upper():
                                if 'Current Ratio' in str(ratio_name):
                                    ratios.current_ratio = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Quick Ratio' in str(ratio_name):
                                    ratios.quick_ratio = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Cash Ratio' in str(ratio_name):
                                    ratios.cash_ratio = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                    
                    # Extract solvency ratios
                    if solvency is not None and not solvency.empty:
                        for ratio_name, ratio_value in solvency.items():
                            if symbol.upper() in str(ratio_name).upper():
                                if 'Debt to Equity Ratio' in str(ratio_name):
                                    ratios.debt_to_equity = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Debt to Assets Ratio' in str(ratio_name):
                                    ratios.debt_to_assets = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Interest Coverage Ratio' in str(ratio_name):
                                    ratios.interest_coverage_ratio = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                    
                    # Extract efficiency ratios
                    if efficiency is not None and not efficiency.empty:
                        for ratio_name, ratio_value in efficiency.items():
                            if symbol.upper() in str(ratio_name).upper():
                                if 'Asset Turnover Ratio' in str(ratio_name):
                                    ratios.asset_turnover = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Inventory Turnover Ratio' in str(ratio_name):
                                    ratios.inventory_turnover = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Receivables Turnover Ratio' in str(ratio_name):
                                    ratios.receivables_turnover = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                    
                    # Extract valuation ratios
                    if valuation is not None and not valuation.empty:
                        for ratio_name, ratio_value in valuation.items():
                            if symbol.upper() in str(ratio_name).upper():
                                if 'Price-to-Earnings Ratio' in str(ratio_name):
                                    ratios.price_to_earnings = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Price-to-Book Ratio' in str(ratio_name):
                                    ratios.price_to_book = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'Price-to-Sales Ratio' in str(ratio_name):
                                    ratios.price_to_sales = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                                elif 'EV/EBITDA' in str(ratio_name):
                                    ratios.enterprise_value_to_ebitda = float(ratio_value.iloc[-1]) if not pd.isna(ratio_value.iloc[-1]) else None
                    
                    results[symbol] = ratios
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to calculate financial ratios: {e}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get financial ratios: {e}")
            return {}
    
    def get_historical_analysis(self, symbols: Union[str, List[str]], 
                              start_date: str = "2023-01-01",
                              end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get historical data with returns and volatility analysis"""
        if not self.available:
            return None
        
        try:
            toolkit = self.create_toolkit(symbols, start_date, end_date, quarterly=False)
            if not toolkit:
                return None
            
            # Get historical data with calculated returns and volatility
            historical = toolkit.get_historical_data()
            return historical
            
        except Exception as e:
            logger.error(f"Failed to get historical analysis: {e}")
            return None

# ============================================================================
# Enhanced FinanceDatabase Adapter
# ============================================================================

class EnhancedFinanceDatabaseAdapter:
    """Enhanced adapter for FinanceDatabase with better search capabilities"""
    
    def __init__(self, cache, rate_limiter):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.available = FINANCE_DATABASE_AVAILABLE
        
        if not self.available:
            logger.warning("FinanceDatabase not available")
            return
        
        # Initialize database objects
        try:
            if FINANCE_DATABASE_AVAILABLE:
                import financedatabase as fd
                self.equities = fd.Equities()
                self.etfs = fd.ETFs()
                self.funds = fd.Funds()
                self.indices = fd.Indices()
                self.currencies = fd.Currencies()
                # Note: Cryptocurrencies may not be available in all versions
                try:
                    # Check if the class exists before using it
                    if hasattr(fd, 'Cryptocurrencies'):
                        self.cryptocurrencies = fd.Cryptocurrencies()
                    else:
                        self.cryptocurrencies = None
                        logger.info("Cryptocurrencies not available in this FinanceDatabase version")
                except (AttributeError, Exception):
                    self.cryptocurrencies = None
                    logger.info("Cryptocurrencies not available in this FinanceDatabase version")
            else:
                self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize FinanceDatabase: {e}")
            self.available = False
    
    def search_securities(self, query: Optional[str] = None, 
                         asset_class: Optional[str] = None,
                         **filters) -> Dict[str, List[SecurityInfo]]:
        """Enhanced security search with unified results"""
        if not self.available:
            return {}
        
        try:
            results = {}
            
            # Search equities
            if asset_class is None or asset_class.lower() == 'equity':
                equities_data = self._search_equities(query, **filters)
                if equities_data:
                    results['equities'] = equities_data
            
            # Search ETFs
            if asset_class is None or asset_class.lower() == 'etf':
                etfs_data = self._search_etfs(query, **filters)
                if etfs_data:
                    results['etfs'] = etfs_data
            
            # Search Funds
            if asset_class is None or asset_class.lower() == 'fund':
                funds_data = self._search_funds(query, **filters)
                if funds_data:
                    results['funds'] = funds_data
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search securities: {e}")
            return {}
    
    def _search_equities(self, query: Optional[str] = None, **filters) -> List[SecurityInfo]:
        """Search equities with enhanced filtering"""
        try:
            if query:
                # Search by name
                df = self.equities.search(name=query, **filters)
            else:
                # Filter by criteria
                df = self.equities.select(**filters)
            
            if df.empty:
                return []
            
            securities = []
            for symbol, row in df.head(50).iterrows():  # Limit to 50 results
                symbol_str = str(symbol)  # Convert to string for type safety
                security = SecurityInfo(
                    symbol=symbol_str,
                    name=row.get('name', ''),
                    asset_class='equity',
                    sector=row.get('sector'),
                    industry=row.get('industry'),
                    country=row.get('country'),
                    exchange=row.get('exchange'),
                    currency=row.get('currency'),
                    market_cap_group=row.get('market_cap'),
                    figi=row.get('figi'),
                    composite_figi=row.get('composite_figi'),
                    shareclass_figi=row.get('shareclass_figi')
                )
                securities.append(security)
            
            return securities
            
        except Exception as e:
            logger.error(f"Failed to search equities: {e}")
            return []
    
    def _search_etfs(self, query: Optional[str] = None, **filters) -> List[SecurityInfo]:
        """Search ETFs with enhanced filtering"""
        try:
            # Fix the category group filter for ETFs
            if 'category_group' in filters:
                # Map common terms to actual ETF category groups
                category_mapping = {
                    'technology': 'Equities',
                    'tech': 'Equities',
                    'equity': 'Equities',
                    'bond': 'Fixed Income',
                    'commodity': 'Commodities',
                    'real estate': 'Real Estate'
                }
                category = filters['category_group'].lower()
                if category in category_mapping:
                    filters['category_group'] = category_mapping[category]
            
            if query:
                # Search by name
                df = self.etfs.search(name=query, **filters)
            else:
                # Filter by criteria
                df = self.etfs.select(**filters)
            
            if df.empty:
                return []
            
            securities = []
            for symbol, row in df.head(50).iterrows():
                symbol_str = str(symbol)  # Convert to string for type safety
                security = SecurityInfo(
                    symbol=symbol_str,
                    name=row.get('name', ''),
                    asset_class='etf',
                    category_group=row.get('category_group'),
                    category=row.get('category'),
                    family=row.get('family'),
                    exchange=row.get('exchange'),
                    currency=row.get('currency'),
                    expense_ratio=row.get('expense_ratio')
                )
                securities.append(security)
            
            return securities
            
        except Exception as e:
            logger.error(f"Failed to search ETFs: {e}")
            return []
    
    def _search_funds(self, query: Optional[str] = None, **filters) -> List[SecurityInfo]:
        """Search funds with enhanced filtering"""
        try:
            if query:
                df = self.funds.search(name=query, **filters)
            else:
                df = self.funds.select(**filters)
            
            if df.empty:
                return []
            
            securities = []
            for symbol, row in df.head(50).iterrows():
                symbol_str = str(symbol)  # Convert to string for type safety
                security = SecurityInfo(
                    symbol=symbol_str,
                    name=row.get('name', ''),
                    asset_class='fund',
                    category_group=row.get('category_group'),
                    category=row.get('category'),
                    family=row.get('family'),
                    exchange=row.get('exchange'),
                    currency=row.get('currency')
                )
                securities.append(security)
            
            return securities
            
        except Exception as e:
            logger.error(f"Failed to search funds: {e}")
            return []
    
    def get_security_info(self, symbol: str) -> Optional[SecurityInfo]:
        """Get detailed information for a specific security"""
        try:
            # Try different asset classes
            for db, asset_class in [
                (self.equities, 'equity'),
                (self.etfs, 'etf'),
                (self.funds, 'fund')
            ]:
                try:
                    df = db.select()
                    if symbol in df.index:
                        row = df.loc[symbol]
                        return SecurityInfo(
                            symbol=symbol,
                            name=row.get('name', ''),
                            asset_class=asset_class,
                            sector=row.get('sector'),
                            industry=row.get('industry'),
                            country=row.get('country'),
                            exchange=row.get('exchange'),
                            currency=row.get('currency'),
                            category_group=row.get('category_group'),
                            category=row.get('category'),
                            family=row.get('family'),
                            market_cap_group=row.get('market_cap'),
                            expense_ratio=row.get('expense_ratio'),
                            figi=row.get('figi'),
                            composite_figi=row.get('composite_figi'),
                            shareclass_figi=row.get('shareclass_figi')
                        )
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get security info for {symbol}: {e}")
            return None
