"""
Enhanced Consolidated Data Feed Integration
Integrates FinanceToolkit, FinanceDatabase, and Enhanced FMP capabilities into the existing consolidated_data_feed

Change note: Removed sys.path manipulation and switched to absolute imports from data_feeds.consolidated_data_feed to avoid ambiguity.
"""

import pandas as pd
from datetime import datetime

from data_feeds.consolidated_data_feed import ConsolidatedDataFeed, DataSource
from data_feeds.finance_integration import (
    FinanceToolkitAdapter,
    EnhancedFinanceDatabaseAdapter,
    FinancialRatios,
    SecurityInfo,
    FINANCE_TOOLKIT_AVAILABLE,
    FINANCE_DATABASE_AVAILABLE
)
from data_feeds.enhanced_fmp_integration import EnhancedFMPAdapter
import logging

logger = logging.getLogger(__name__)

class EnhancedConsolidatedDataFeed(ConsolidatedDataFeed):
    """Enhanced version of ConsolidatedDataFeed with FinanceToolkit, FinanceDatabase, and Enhanced FMP integration"""
    
    def __init__(self):
        # Initialize parent class
        super().__init__()
        
        # Add new adapters
        if FINANCE_TOOLKIT_AVAILABLE:
            self.finance_toolkit = FinanceToolkitAdapter(self.cache, self.rate_limiter)
            logger.info("FinanceToolkit adapter initialized")
        else:
            self.finance_toolkit = None
            logger.warning("FinanceToolkit not available")
        
        if FINANCE_DATABASE_AVAILABLE:
            self.enhanced_finance_db = EnhancedFinanceDatabaseAdapter(self.cache, self.rate_limiter)
            logger.info("Enhanced FinanceDatabase adapter initialized")
        else:
            self.enhanced_finance_db = None
            logger.warning("Enhanced FinanceDatabase not available")
        
        # Add Enhanced FMP adapter
        self.enhanced_fmp = EnhancedFMPAdapter()
        if self.enhanced_fmp.api_key:
            logger.info("Enhanced FMP adapter initialized with API key")
        else:
            logger.warning("Enhanced FMP adapter initialized without API key - some features limited")
    
    def get_enhanced_financial_ratios(self, symbols, period="annual"):
        """Get comprehensive financial ratios using Enhanced FMP"""
        if not self.enhanced_fmp.api_key:
            logger.warning("Enhanced FMP features require API key")
            return {}
        
        results = {}
        if isinstance(symbols, str):
            symbols = [symbols]
        
        for symbol in symbols:
            try:
                ratios = self.enhanced_fmp.get_financial_ratios(symbol, period)
                if ratios:
                    results[symbol] = ratios[0]  # Get most recent ratios
                    logger.info(f"Retrieved enhanced financial ratios for {symbol}")
                else:
                    logger.warning(f"No enhanced financial ratios found for {symbol}")
            except Exception as e:
                logger.error(f"Error getting enhanced financial ratios for {symbol}: {e}")
        
        return results
    
    def get_dcf_valuations(self, symbols):
        """Get DCF valuations using Enhanced FMP"""
        if not self.enhanced_fmp.api_key:
            logger.warning("Enhanced FMP features require API key")
            return {}
        
        results = {}
        if isinstance(symbols, str):
            symbols = [symbols]
        
        for symbol in symbols:
            try:
                dcf = self.enhanced_fmp.get_dcf_valuation(symbol)
                if dcf:
                    results[symbol] = dcf[0] if isinstance(dcf, list) else dcf
                    logger.info(f"Retrieved DCF valuation for {symbol}")
            except Exception as e:
                logger.error(f"Error getting DCF valuation for {symbol}: {e}")
        
        return results
    
    def get_comprehensive_fundamentals(self, symbol, period="annual"):
        """Get comprehensive fundamental data using Enhanced FMP"""
        if not self.enhanced_fmp.api_key:
            logger.warning("Enhanced FMP features require API key")
            return {}
        
        fundamentals = {}
        
        try:
            # Get financial statements
            income_statement = self.enhanced_fmp.get_income_statement(symbol, period, limit=3)
            balance_sheet = self.enhanced_fmp.get_balance_sheet(symbol, period, limit=3)
            cash_flow = self.enhanced_fmp.get_cash_flow(symbol, period, limit=3)
            key_metrics = self.enhanced_fmp.get_key_metrics(symbol, period, limit=3)
            
            fundamentals[symbol] = {
                'income_statement': income_statement,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'key_metrics': key_metrics
            }
            
            logger.info(f"Retrieved comprehensive fundamentals for {symbol}")
            
        except Exception as e:
            logger.error(f"Error getting comprehensive fundamentals for {symbol}: {e}")
        
        return fundamentals
    
    def get_analyst_data(self, symbol):
        """Get analyst estimates and price targets"""
        if not self.enhanced_fmp.api_key:
            logger.warning("Enhanced FMP features require API key")
            return {}
        
        analyst_data = {}
        
        try:
            estimates = self.enhanced_fmp.get_analyst_estimates(symbol)
            price_target = self.enhanced_fmp.get_price_target(symbol)
            
            analyst_data[symbol] = {
                'estimates': estimates,
                'price_target': price_target
            }
            
            logger.info(f"Retrieved analyst data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error getting analyst data for {symbol}: {e}")
        
        return analyst_data
    
    def get_institutional_data(self, symbol):
        """Get institutional ownership and insider trading data"""
        if not self.enhanced_fmp.api_key:
            logger.warning("Enhanced FMP features require API key")
            return {}
        
        institutional_data = {}
        
        try:
            ownership = self.enhanced_fmp.get_institutional_ownership(symbol)
            insider_trading = self.enhanced_fmp.get_insider_trading(symbol)
            
            institutional_data[symbol] = {
                'institutional_ownership': ownership,
                'insider_trading': insider_trading
            }
            
            logger.info(f"Retrieved institutional data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error getting institutional data for {symbol}: {e}")
        
        return institutional_data
    
    def get_market_analysis(self):
        """Get comprehensive market analysis data"""
        if not self.enhanced_fmp.api_key:
            logger.warning("Enhanced FMP features require API key")
            return {}
        
        market_data = {}
        
        try:
            sector_performance = self.enhanced_fmp.get_sector_performance()
            market_cap_ranking = self.enhanced_fmp.get_market_cap_ranking()
            
            market_data = {
                'sector_performance': sector_performance,
                'market_cap_ranking': market_cap_ranking
            }
            
            logger.info("Retrieved market analysis data")
            
        except Exception as e:
            logger.error(f"Error getting market analysis data: {e}")
        
        return market_data
    
    def search_and_screen_stocks(self, query=None, **screening_criteria):
        """Search for companies and screen stocks using Enhanced FMP"""
        if not self.enhanced_fmp.api_key:
            logger.warning("Enhanced FMP features require API key")
            return {}
        
        results = {}
        
        try:
            if query:
                search_results = self.enhanced_fmp.search_companies(query)
                results['search'] = search_results
            
            if screening_criteria:
                screening_results = self.enhanced_fmp.get_stock_screener(**screening_criteria)
                results['screening'] = screening_results
            
            logger.info("Retrieved stock search and screening results")
            
        except Exception as e:
            logger.error(f"Error in stock search/screening: {e}")
        
        return results
    
    def get_financial_ratios(self, symbols, period="annual"):
        """Get comprehensive financial ratios using FinanceToolkit"""
        if not self.finance_toolkit or not self.finance_toolkit.available:
            logger.warning("FinanceToolkit not available for financial ratios")
            return {}
        
        try:
            return self.finance_toolkit.get_financial_ratios(symbols, period)
        except Exception as e:
            logger.error(f"Failed to get financial ratios: {e}")
            return {}
    
    def get_historical_analysis(self, symbols, start_date="2023-01-01", end_date=None):
        """Get historical data with returns and volatility analysis using FinanceToolkit"""
        if not self.finance_toolkit or not self.finance_toolkit.available:
            logger.warning("FinanceToolkit not available for historical analysis")
            return None
        
        try:
            return self.finance_toolkit.get_historical_analysis(symbols, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get historical analysis: {e}")
            return None
    
    def search_enhanced_securities(self, query=None, asset_class=None, **filters):
        """Enhanced security search using FinanceDatabase"""
        if not self.enhanced_finance_db or not self.enhanced_finance_db.available:
            logger.warning("Enhanced FinanceDatabase not available for security search")
            return {}
        
        try:
            return self.enhanced_finance_db.search_securities(query, asset_class, **filters)
        except Exception as e:
            logger.error(f"Failed to search securities: {e}")
            return {}
    
    def get_enhanced_security_info(self, symbol):
        """Get detailed security information using FinanceDatabase"""
        if not self.enhanced_finance_db or not self.enhanced_finance_db.available:
            logger.warning("Enhanced FinanceDatabase not available for security info")
            return None
        
        try:
            return self.enhanced_finance_db.get_security_info(symbol)
        except Exception as e:
            logger.error(f"Failed to get enhanced security info: {e}")
            return None
    
    def get_comprehensive_analysis(self, symbol):
        """Get comprehensive analysis combining all data sources"""
        results = {
            'symbol': symbol,
            'timestamp': None,
            'quote': None,
            'company_info': None,
            'enhanced_security_info': None,
            'financial_ratios': None,
            'historical_analysis': None,
            'news': []
        }
        
        try:
            # Get basic quote
            results['quote'] = self.get_quote(symbol)
            
            # Get company information
            results['company_info'] = self.get_company_info(symbol)
            
            # Get enhanced security information
            results['enhanced_security_info'] = self.get_enhanced_security_info(symbol)
            
            # Get financial ratios
            ratios = self.get_financial_ratios([symbol])
            if ratios and symbol in ratios:
                results['financial_ratios'] = ratios[symbol]
            
            # Get historical analysis (last 1 year)
            results['historical_analysis'] = self.get_historical_analysis([symbol])
            
            # Get news
            results['news'] = self.get_news(symbol, limit=5)
            
            results['timestamp'] = pd.Timestamp.now()
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive analysis for {symbol}: {e}")
        
        return results
    
    def screen_securities(self, criteria):
        """Screen securities based on criteria"""
        try:
            # Example criteria:
            # {
            #     'sector': 'Information Technology',
            #     'market_cap': 'Large Cap',
            #     'country': 'United States',
            #     'min_market_cap': 10000000000,  # $10B
            #     'max_pe_ratio': 30,
            #     'min_profit_margin': 0.1  # 10%
            # }
            
            # Step 1: Find securities using FinanceDatabase
            db_criteria = {k: v for k, v in criteria.items() 
                          if k in ['sector', 'market_cap', 'country', 'industry']}
            
            securities = self.search_enhanced_securities(asset_class='equity', **db_criteria)
            
            if not securities.get('equities'):
                return []
            
            # Step 2: Get financial data for filtering
            symbols = [sec.symbol for sec in securities['equities'][:50]]  # Limit to 50 for performance
            results = []
            
            for symbol in symbols:
                try:
                    # Get quote for market cap and PE ratio
                    quote = self.get_quote(symbol)
                    if not quote:
                        continue
                    
                    # Apply financial filters
                    if 'min_market_cap' in criteria:
                        if not quote.market_cap or quote.market_cap < criteria['min_market_cap']:
                            continue
                    
                    if 'max_pe_ratio' in criteria:
                        if not quote.pe_ratio or quote.pe_ratio > criteria['max_pe_ratio']:
                            continue
                    
                    # Get financial ratios if needed
                    ratios_data = None
                    if 'min_profit_margin' in criteria:
                        ratios_data = self.get_financial_ratios([symbol])
                        if ratios_data and symbol in ratios_data:
                            profit_margin = ratios_data[symbol].net_profit_margin
                            if not profit_margin or profit_margin < criteria['min_profit_margin']:
                                continue
                    
                    # If we get here, the security passed all filters
                    security_info = next((s for s in securities['equities'] if s.symbol == symbol), None)
                    results.append({
                        'security': security_info,
                        'quote': quote,
                        'ratios': ratios_data.get(symbol) if ratios_data else None
                    })
                    
                except Exception as e:
                    logger.warning(f"Error screening {symbol}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Security screening failed: {e}")
            return []
    
    def get_sector_analysis(self, sector="Information Technology", limit=10):
        """Get analysis for top companies in a sector"""
        try:
            # Find companies in the sector
            securities = self.search_enhanced_securities(
                asset_class='equity',
                sector=sector,
                country='United States',
                market_cap='Large Cap'
            )
            
            if not securities.get('equities'):
                return {}
            
            # Get top companies (limit the analysis for performance)
            top_companies = securities['equities'][:limit]
            symbols = [company.symbol for company in top_companies]
            
            # Get financial ratios for comparison
            ratios = self.get_financial_ratios(symbols)
            
            # Get quotes for current prices
            quotes = {}
            for symbol in symbols:
                quote = self.get_quote(symbol)
                if quote:
                    quotes[symbol] = quote
            
            # Compile results
            results = {
                'sector': sector,
                'companies': [],
                'sector_averages': {}
            }
            
            # Collect individual company data
            for company in top_companies:
                symbol = company.symbol
                company_data = {
                    'security': company,
                    'quote': quotes.get(symbol),
                    'ratios': ratios.get(symbol)
                }
                results['companies'].append(company_data)
            
            # Calculate sector averages
            if ratios:
                ratio_values = {
                    'profit_margin': [],
                    'roe': [],
                    'current_ratio': [],
                    'pe_ratio': []
                }
                
                for symbol, ratio in ratios.items():
                    if ratio.net_profit_margin:
                        ratio_values['profit_margin'].append(ratio.net_profit_margin)
                    if ratio.return_on_equity:
                        ratio_values['roe'].append(ratio.return_on_equity)
                    if ratio.current_ratio:
                        ratio_values['current_ratio'].append(ratio.current_ratio)
                    if quotes.get(symbol) and quotes[symbol].pe_ratio:
                        ratio_values['pe_ratio'].append(float(quotes[symbol].pe_ratio))
                
                # Calculate averages
                for metric, values in ratio_values.items():
                    if values:
                        results['sector_averages'][metric] = sum(values) / len(values)
            
            return results
            
        except Exception as e:
            logger.error(f"Sector analysis failed: {e}")
            return {}

# ============================================================================
# Convenience Functions for Enhanced Features
# ============================================================================

def create_enhanced_data_feed():
    """Create an enhanced data feed instance"""
    return EnhancedConsolidatedDataFeed()

def quick_analysis(symbol):
    """Quick comprehensive analysis of a symbol"""
    feed = create_enhanced_data_feed()
    return feed.get_comprehensive_analysis(symbol)

def screen_tech_stocks(min_market_cap=10_000_000_000, max_pe=30, min_margin=0.1):
    """Screen technology stocks with specific criteria"""
    feed = create_enhanced_data_feed()
    return feed.screen_securities({
        'sector': 'Information Technology',
        'market_cap': 'Large Cap',
        'country': 'United States',
        'min_market_cap': min_market_cap,
        'max_pe_ratio': max_pe,
        'min_profit_margin': min_margin
    })

def analyze_sector(sector="Information Technology", limit=10):
    """Analyze top companies in a sector"""
    feed = create_enhanced_data_feed()
    return feed.get_sector_analysis(sector, limit)

# Global enhanced instance
_enhanced_data_feed = None

def get_enhanced_data_feed():
    """Get the global enhanced data feed instance"""
    global _enhanced_data_feed
    if _enhanced_data_feed is None:
        _enhanced_data_feed = create_enhanced_data_feed()
    return _enhanced_data_feed
