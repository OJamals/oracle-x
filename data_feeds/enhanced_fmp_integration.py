"""
Enhanced Financial Modeling Prep (FMP) Integration
Provides comprehensive financial data integration with FMP API including:
- Real-time quotes and market data
- Company profiles and fundamental data
- Financial statements and ratios
- Economic indicators and market analysis
- Options data and institutional ownership
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class FMPFinancialRatios:
    """Financial ratios from FMP"""
    symbol: str
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    gross_profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_profit_margin: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    price_to_sales: Optional[float] = None
    peg_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

@dataclass
class FMPEconomicIndicator:
    """Economic indicator from FMP"""
    indicator: str
    value: float
    date: datetime
    country: Optional[str] = None

@dataclass
class FMPEarningsCalendar:
    """Earnings calendar entry from FMP"""
    symbol: str
    date: datetime
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    time: Optional[str] = None  # 'bmo' (before market open) or 'amc' (after market close)

@dataclass
class FMPInsiderTrading:
    """Insider trading data from FMP"""
    symbol: str
    reporting_name: str
    relationship: str
    transaction_date: datetime
    transaction_type: str
    securities_owned: Optional[int] = None
    securities_transacted: Optional[int] = None
    price: Optional[float] = None

class EnhancedFMPAdapter:
    """Enhanced Financial Modeling Prep API adapter with comprehensive features"""
    
    def __init__(self):
        # Try to get API key from configuration manager first, then fallback to environment
        try:
            from config_manager import get_financial_modeling_prep_api_key
            self.api_key = get_financial_modeling_prep_api_key()
        except ImportError:
            self.api_key = os.getenv('FINANCIALMODELINGPREP_API_KEY')

        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.base_url_v4 = "https://financialmodelingprep.com/api/v4"

        if not self.api_key:
            logger.warning("FMP API key not found. Some features may be limited.")
    
    def _make_request(self, endpoint: str, api_version: str = "v3") -> Optional[Any]:
        """Make authenticated request to FMP API"""
        if not self.api_key:
            logger.error("FMP API key required for this operation")
            return None
        
        try:
            base_url = self.base_url if api_version == "v3" else self.base_url_v4
            url = f"{base_url}/{endpoint}"
            
            # Add API key as parameter
            separator = "&" if "?" in endpoint else "?"
            url = f"{url}{separator}apikey={self.api_key}"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 429:
                logger.warning("FMP API rate limit exceeded")
                return None
            else:
                logger.error(f"FMP API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"FMP API request error: {e}")
            return None
    
    def get_financial_ratios(self, symbol: str, period: str = "annual") -> Optional[List[FMPFinancialRatios]]:
        """Get comprehensive financial ratios for a symbol"""
        endpoint = f"ratios/{symbol}?period={period}&limit=5"
        data = self._make_request(endpoint)
        
        if not data:
            return None
        
        ratios = []
        for item in data:
            ratios.append(FMPFinancialRatios(
                symbol=symbol,
                current_ratio=item.get('currentRatio'),
                quick_ratio=item.get('quickRatio'),
                debt_to_equity=item.get('debtEquityRatio'),
                roe=item.get('returnOnEquity'),
                roa=item.get('returnOnAssets'),
                gross_profit_margin=item.get('grossProfitMargin'),
                operating_margin=item.get('operatingProfitMargin'),
                net_profit_margin=item.get('netProfitMargin'),
                pe_ratio=item.get('priceEarningsRatio'),
                pb_ratio=item.get('priceToBookRatio'),
                price_to_sales=item.get('priceToSalesRatio'),
                peg_ratio=item.get('pegRatio'),
                dividend_yield=item.get('dividendYield'),
                payout_ratio=item.get('payoutRatio')
            ))
        
        return ratios
    
    def get_income_statement(self, symbol: str, period: str = "annual", limit: int = 5) -> Optional[pd.DataFrame]:
        """Get income statement data"""
        endpoint = f"income-statement/{symbol}?period={period}&limit={limit}"
        data = self._make_request(endpoint)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                return df.sort_values('date', ascending=False)
        
        return None
    
    def get_balance_sheet(self, symbol: str, period: str = "annual", limit: int = 5) -> Optional[pd.DataFrame]:
        """Get balance sheet data"""
        endpoint = f"balance-sheet-statement/{symbol}?period={period}&limit={limit}"
        data = self._make_request(endpoint)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                return df.sort_values('date', ascending=False)
        
        return None
    
    def get_cash_flow(self, symbol: str, period: str = "annual", limit: int = 5) -> Optional[pd.DataFrame]:
        """Get cash flow statement data"""
        endpoint = f"cash-flow-statement/{symbol}?period={period}&limit={limit}"
        data = self._make_request(endpoint)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                return df.sort_values('date', ascending=False)
        
        return None
    
    def get_key_metrics(self, symbol: str, period: str = "annual", limit: int = 5) -> Optional[pd.DataFrame]:
        """Get key financial metrics"""
        endpoint = f"key-metrics/{symbol}?period={period}&limit={limit}"
        data = self._make_request(endpoint)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                return df.sort_values('date', ascending=False)
        
        return None
    
    def get_dcf_valuation(self, symbol: str) -> Optional[Dict]:
        """Get DCF valuation analysis"""
        endpoint = f"discounted-cash-flow/{symbol}"
        return self._make_request(endpoint)
    
    def get_earnings_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> Optional[List[FMPEarningsCalendar]]:
        """Get earnings calendar"""
        if not from_date:
            from_date = datetime.now().strftime('%Y-%m-%d')
        if not to_date:
            to_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        
        endpoint = f"earning_calendar?from={from_date}&to={to_date}"
        data = self._make_request(endpoint)
        
        if not data:
            return None
        
        calendar = []
        for item in data:
            calendar.append(FMPEarningsCalendar(
                symbol=item.get('symbol', ''),
                date=datetime.strptime(item.get('date', ''), '%Y-%m-%d') if item.get('date') else datetime.now(),
                eps_estimate=item.get('epsEstimate'),
                eps_actual=item.get('eps'),
                revenue_estimate=item.get('revenueEstimate'),
                revenue_actual=item.get('revenue'),
                time=item.get('time')
            ))
        
        return calendar
    
    def get_economic_indicators(self, indicator: str) -> Optional[List[FMPEconomicIndicator]]:
        """Get economic indicators (GDP, inflation, unemployment, etc.)"""
        endpoint = f"economic?name={indicator}"
        data = self._make_request(endpoint)
        
        if not data:
            return None
        
        indicators = []
        for item in data:
            indicators.append(FMPEconomicIndicator(
                indicator=indicator,
                value=item.get('value', 0),
                date=datetime.strptime(item.get('date', ''), '%Y-%m-%d') if item.get('date') else datetime.now()
            ))
        
        return indicators
    
    def get_insider_trading(self, symbol: str, limit: int = 50) -> Optional[List[FMPInsiderTrading]]:
        """Get insider trading data"""
        endpoint = f"insider-trading?symbol={symbol}&limit={limit}"
        data = self._make_request(endpoint, api_version="v4")
        
        if not data:
            return None
        
        insider_trades = []
        for item in data:
            insider_trades.append(FMPInsiderTrading(
                symbol=symbol,
                reporting_name=item.get('reportingName', ''),
                relationship=item.get('relationship', ''),
                transaction_date=datetime.strptime(item.get('transactionDate', ''), '%Y-%m-%d') if item.get('transactionDate') else datetime.now(),
                transaction_type=item.get('transactionType', ''),
                securities_owned=item.get('securitiesOwned'),
                securities_transacted=item.get('securitiesTransacted'),
                price=item.get('price')
            ))
        
        return insider_trades
    
    def get_institutional_ownership(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get institutional ownership data"""
        endpoint = f"institutional-holder/{symbol}"
        data = self._make_request(endpoint)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                return df
        
        return None
    
    def get_analyst_estimates(self, symbol: str, period: str = "annual") -> Optional[pd.DataFrame]:
        """Get analyst earnings and revenue estimates"""
        endpoint = f"analyst-estimates/{symbol}?period={period}"
        data = self._make_request(endpoint)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                return df.sort_values('date', ascending=False)
        
        return None
    
    def get_price_target(self, symbol: str) -> Optional[Dict]:
        """Get analyst price targets"""
        endpoint = f"price-target?symbol={symbol}"
        data = self._make_request(endpoint)
        
        if data and len(data) > 0:
            return data[0]  # Return most recent price target
        
        return None
    
    def get_market_cap_ranking(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get market cap rankings"""
        endpoint = f"stock_market/gainers?limit={limit}"
        data = self._make_request(endpoint)
        
        if data:
            df = pd.DataFrame(data)
            return df
        
        return None
    
    def get_sector_performance(self) -> Optional[pd.DataFrame]:
        """Get sector performance data"""
        endpoint = "sectors-performance"
        data = self._make_request(endpoint)
        
        if data:
            df = pd.DataFrame(data)
            return df
        
        return None
    
    def get_crypto_data(self, symbol: str = "BTCUSD") -> Optional[Dict]:
        """Get cryptocurrency data"""
        endpoint = f"quote/{symbol}"
        return self._make_request(endpoint)
    
    def get_forex_data(self, pair: str = "EURUSD") -> Optional[Dict]:
        """Get forex data"""
        endpoint = f"fx/{pair}"
        return self._make_request(endpoint)
    
    def search_companies(self, query: str, limit: int = 10) -> Optional[List[Dict]]:
        """Search for companies"""
        endpoint = f"search?query={query}&limit={limit}"
        return self._make_request(endpoint)
    
    def get_stock_screener(self, market_cap_more_than: int = 1000000000, 
                          beta_more_than: float = 0, beta_lower_than: float = 2,
                          volume_more_than: int = 10000, limit: int = 50) -> Optional[List[Dict]]:
        """Screen stocks based on criteria"""
        endpoint = f"stock-screener?marketCapMoreThan={market_cap_more_than}&betaMoreThan={beta_more_than}&betaLowerThan={beta_lower_than}&volumeMoreThan={volume_more_than}&limit={limit}"
        return self._make_request(endpoint)

# Integration function for enhanced FMP features
def integrate_enhanced_fmp_features():
    """Integrate enhanced FMP features into the trading system"""
    fmp = EnhancedFMPAdapter()
    
    if not fmp.api_key:
        logger.error("FMP API key required for enhanced features")
        return None
    
    logger.info("Enhanced FMP adapter initialized successfully")
    return fmp

if __name__ == "__main__":
    # Test the enhanced FMP integration
    fmp = EnhancedFMPAdapter()
    
    if fmp.api_key:
        print("Testing Enhanced FMP Features...")
        
        # Test financial ratios
        print("\n1. Testing Financial Ratios for AAPL:")
        ratios = fmp.get_financial_ratios("AAPL")
        if ratios:
            latest = ratios[0]
            print(f"   PE Ratio: {latest.pe_ratio}")
            print(f"   ROE: {latest.roe:.2%}" if latest.roe else "   ROE: N/A")
            print(f"   Debt/Equity: {latest.debt_to_equity}" if latest.debt_to_equity else "   Debt/Equity: N/A")
        
        # Test DCF valuation
        print("\n2. Testing DCF Valuation for AAPL:")
        dcf = fmp.get_dcf_valuation("AAPL")
        if dcf and len(dcf) > 0:
            valuation = dcf[0]
            print(f"   DCF Value: ${valuation.get('dcf', 'N/A')}")
            print(f"   Current Price: ${valuation.get('Stock Price', 'N/A')}")
        
        # Test earnings calendar
        print("\n3. Testing Earnings Calendar:")
        earnings = fmp.get_earnings_calendar()
        if earnings:
            print(f"   Found {len(earnings)} upcoming earnings")
            for earning in earnings[:3]:  # Show first 3
                print(f"   {earning.symbol}: {earning.date.strftime('%Y-%m-%d')} ({earning.time})")
        
        print("\n✅ Enhanced FMP integration test completed!")
    else:
        print("❌ FMP API key not found - please set FINANCIALMODELINGPREP_API_KEY")
