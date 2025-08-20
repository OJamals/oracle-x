"""
Comprehensive test suite for the Options Valuation Engine
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Import the module to test
from data_feeds.options_valuation_engine import (
    OptionsValuationEngine,
    OptionContract,
    OptionType,
    OptionStyle,
    PricingModel,
    ValuationResult,
    OpportunityAnalysis,
    IVSurfacePoint,
    BinomialModel,
    MonteCarloModel,
    create_valuation_engine,
    analyze_options_chain
)


class TestOptionContract(unittest.TestCase):
    """Test OptionContract data class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.expiry = datetime.now() + timedelta(days=30)
        self.contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=self.expiry,
            option_type=OptionType.CALL,
            style=OptionStyle.AMERICAN,
            bid=5.0,
            ask=5.5,
            last=5.25,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            underlying_price=155.0
        )
    
    def test_market_price_calculation(self):
        """Test market price calculation"""
        # Test with bid and ask
        self.assertEqual(self.contract.market_price, 5.25)
        
        # Test with only last price
        contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=self.expiry,
            option_type=OptionType.CALL,
            last=5.0
        )
        self.assertEqual(contract.market_price, 5.0)
        
        # Test with no prices
        contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=self.expiry,
            option_type=OptionType.CALL
        )
        self.assertIsNone(contract.market_price)
    
    def test_time_to_expiry(self):
        """Test time to expiry calculation"""
        tte = self.contract.time_to_expiry
        self.assertAlmostEqual(tte, 30/365, places=2)
        
        # Test expired option
        expired_contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=datetime.now() - timedelta(days=1),
            option_type=OptionType.CALL
        )
        self.assertEqual(expired_contract.time_to_expiry, 0)


class TestBinomialModel(unittest.TestCase):
    """Test Binomial pricing model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = BinomialModel(steps=50)
    
    def test_european_call_pricing(self):
        """Test European call option pricing"""
        price = self.model.price(
            S=100, K=100, r=0.05, q=0.02, sigma=0.25, T=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )
        # Should be close to Black-Scholes price
        self.assertGreater(price, 0)
        self.assertLess(price, 20)
    
    def test_american_put_pricing(self):
        """Test American put option pricing"""
        price = self.model.price(
            S=100, K=110, r=0.05, q=0.02, sigma=0.25, T=1.0,
            option_type=OptionType.PUT,
            style=OptionStyle.AMERICAN
        )
        # American put should be worth at least intrinsic value
        intrinsic = max(0, 110 - 100)
        self.assertGreaterEqual(price, intrinsic)
    
    def test_expired_option(self):
        """Test pricing of expired option"""
        # ITM call
        price = self.model.price(
            S=110, K=100, r=0.05, q=0.02, sigma=0.25, T=0,
            option_type=OptionType.CALL,
            style=OptionStyle.AMERICAN
        )
        self.assertEqual(price, 10)
        
        # OTM put
        price = self.model.price(
            S=110, K=100, r=0.05, q=0.02, sigma=0.25, T=0,
            option_type=OptionType.PUT,
            style=OptionStyle.AMERICAN
        )
        self.assertEqual(price, 0)


class TestMonteCarloModel(unittest.TestCase):
    """Test Monte Carlo pricing model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = MonteCarloModel(simulations=1000, seed=42)
    
    def test_call_pricing(self):
        """Test call option pricing"""
        price, std_error = self.model.price(
            S=100, K=100, r=0.05, q=0.02, sigma=0.25, T=1.0,
            option_type=OptionType.CALL
        )
        self.assertGreater(price, 0)
        self.assertLess(price, 30)
        self.assertGreater(std_error, 0)
    
    def test_put_pricing(self):
        """Test put option pricing"""
        price, std_error = self.model.price(
            S=100, K=110, r=0.05, q=0.02, sigma=0.25, T=1.0,
            option_type=OptionType.PUT
        )
        self.assertGreater(price, 0)
        self.assertLess(std_error, price * 0.1)  # Standard error should be small
    
    def test_expired_option(self):
        """Test pricing of expired option"""
        price, std_error = self.model.price(
            S=110, K=100, r=0.05, q=0.02, sigma=0.25, T=0,
            option_type=OptionType.CALL
        )
        self.assertEqual(price, 10)
        self.assertEqual(std_error, 0)


class TestOptionsValuationEngine(unittest.TestCase):
    """Test main valuation engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = OptionsValuationEngine(
            risk_free_rate=0.05,
            cache_ttl=60,
            confidence_threshold=0.7,
            max_workers=2
        )
        
        self.expiry = datetime.now() + timedelta(days=30)
        self.contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=self.expiry,
            option_type=OptionType.CALL,
            style=OptionStyle.AMERICAN,
            bid=5.0,
            ask=5.5,
            last=5.25,
            volume=1000,
            open_interest=5000,
            underlying_price=155.0
        )
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.risk_free_rate, 0.05)
        self.assertEqual(self.engine.cache_ttl, 60)
        self.assertEqual(self.engine.confidence_threshold, 0.7)
        self.assertEqual(len(self.engine.models), 3)
        self.assertIn(PricingModel.BLACK_SCHOLES, self.engine.model_weights)
    
    def test_calculate_fair_value(self):
        """Test fair value calculation"""
        fair_value, model_prices = self.engine.calculate_fair_value(
            self.contract,
            underlying_price=155.0,
            volatility=0.25,
            dividend_yield=0.02
        )
        
        self.assertGreater(fair_value, 0)
        self.assertEqual(len(model_prices), 3)
        self.assertIn('black_scholes', model_prices)
        self.assertIn('binomial', model_prices)
        self.assertIn('monte_carlo', model_prices)
    
    def test_caching(self):
        """Test calculation caching"""
        # First call - should calculate
        fair_value1, _ = self.engine.calculate_fair_value(
            self.contract,
            underlying_price=155.0,
            volatility=0.25,
            use_cache=True
        )
        
        # Second call - should use cache
        fair_value2, _ = self.engine.calculate_fair_value(
            self.contract,
            underlying_price=155.0,
            volatility=0.25,
            use_cache=True
        )
        
        self.assertEqual(fair_value1, fair_value2)
        
        # Check cache stats
        stats = self.engine.get_cache_stats()
        self.assertGreater(stats['total_entries'], 0)
    
    def test_detect_mispricing(self):
        """Test mispricing detection"""
        market_data = pd.DataFrame({
            'Close': np.random.normal(155, 5, 100)
        })
        
        result = self.engine.detect_mispricing(
            self.contract,
            underlying_price=155.0,
            market_data=market_data
        )
        
        self.assertIsInstance(result, ValuationResult)
        self.assertEqual(result.contract, self.contract)
        self.assertIsNotNone(result.theoretical_value)
        self.assertIsNotNone(result.mispricing)
        self.assertIsNotNone(result.confidence_score)
        self.assertIn('delta', result.greeks)
    
    def test_undervalued_detection(self):
        """Test undervalued option detection"""
        # Create an undervalued option
        contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=self.expiry,
            option_type=OptionType.CALL,
            bid=2.0,
            ask=2.5,
            volume=1000,
            open_interest=5000,
            underlying_price=155.0
        )
        
        result = self.engine.detect_mispricing(
            contract,
            underlying_price=155.0
        )
        
        # Should detect as undervalued (market price much lower than theoretical)
        if result.theoretical_value > result.market_price * 1.05:
            self.assertTrue(result.is_undervalued)
    
    def test_analyze_iv_surface(self):
        """Test IV surface analysis"""
        options_chain = [
            OptionContract(
                symbol="AAPL",
                strike=strike,
                expiry=self.expiry,
                option_type=OptionType.CALL,
                bid=max(0.5, 155 - strike + 2),
                ask=max(0.5, 155 - strike + 2.5),
                underlying_price=155.0
            )
            for strike in [145, 150, 155, 160, 165]
        ]
        
        surface_points = self.engine.analyze_iv_surface(
            options_chain,
            underlying_price=155.0
        )
        
        self.assertEqual(len(surface_points), 5)
        for point in surface_points:
            self.assertIsInstance(point, IVSurfacePoint)
            self.assertGreater(point.implied_vol, 0)
            self.assertGreater(point.moneyness, 0)
    
    def test_calculate_expected_returns(self):
        """Test expected returns calculation"""
        # Create a valuation result
        valuation = ValuationResult(
            contract=self.contract,
            theoretical_value=6.0,
            market_price=5.25,
            mispricing=0.75,
            mispricing_ratio=0.14,
            confidence_score=80.0,
            pricing_breakdown={'black_scholes': 6.0},
            greeks={'delta': 0.6},
            implied_volatility=0.25
        )
        
        expected_return, prob_profit = self.engine.calculate_expected_returns(
            valuation,
            target_price=160.0,
            probability_model="normal"
        )
        
        self.assertIsInstance(expected_return, float)
        self.assertIsInstance(prob_profit, float)
        self.assertGreaterEqual(prob_profit, 0)
        self.assertLessEqual(prob_profit, 1)
    
    def test_scan_opportunities(self):
        """Test batch opportunity scanning"""
        options_chain = [
            OptionContract(
                symbol="AAPL",
                strike=strike,
                expiry=self.expiry,
                option_type=OptionType.CALL,
                bid=max(0.5, 155 - strike + 1),
                ask=max(0.5, 155 - strike + 1.5),
                volume=1000,
                open_interest=5000,
                underlying_price=155.0
            )
            for strike in [145, 150, 155, 160, 165]
        ]
        
        opportunities = self.engine.scan_opportunities(
            options_chain,
            underlying_price=155.0,
            min_opportunity_score=0  # Accept all for testing
        )
        
        # Should return some opportunities
        self.assertIsInstance(opportunities, list)
        for opp in opportunities:
            self.assertIsInstance(opp, OpportunityAnalysis)
            self.assertGreaterEqual(opp.valuation.opportunity_score, 0)
    
    def test_historical_volatility_calculation(self):
        """Test historical volatility calculation"""
        market_data = pd.DataFrame({
            'Close': [100, 102, 101, 103, 104, 102, 105, 103, 106, 104]
        })
        
        hist_vol = self.engine._calculate_historical_volatility(market_data, periods=5)
        
        self.assertGreater(hist_vol, 0)
        self.assertLess(hist_vol, 2)  # Reasonable volatility range
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation"""
        model_prices = {
            'black_scholes': 5.0,
            'binomial': 5.2,
            'monte_carlo': 4.8
        }
        
        # High liquidity contract
        liquid_contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=self.expiry,
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.1,
            volume=10000,
            open_interest=50000
        )
        
        confidence = self.engine._calculate_confidence_score(
            liquid_contract,
            model_prices,
            market_price=5.0
        )
        
        self.assertGreater(confidence, 70)  # Should have high confidence
        
        # Low liquidity contract
        illiquid_contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=datetime.now() + timedelta(days=2),  # Very short term
            option_type=OptionType.CALL,
            bid=5.0,
            ask=6.0,  # Wide spread
            volume=5,
            open_interest=10
        )
        
        confidence = self.engine._calculate_confidence_score(
            illiquid_contract,
            model_prices,
            market_price=5.5
        )
        
        self.assertLess(confidence, 50)  # Should have low confidence
    
    def test_cache_operations(self):
        """Test cache operations"""
        # Add to cache
        self.engine._set_cache("test_key", "test_value")
        
        # Retrieve from cache
        value = self.engine._get_from_cache("test_key")
        self.assertEqual(value, "test_value")
        
        # Clear cache
        self.engine.clear_cache()
        stats = self.engine.get_cache_stats()
        self.assertEqual(stats['total_entries'], 0)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory and helper functions"""
    
    def test_create_valuation_engine(self):
        """Test engine factory function"""
        # Default configuration
        engine = create_valuation_engine()
        self.assertIsInstance(engine, OptionsValuationEngine)
        self.assertEqual(engine.risk_free_rate, 0.05)
        
        # Custom configuration
        config = {
            'risk_free_rate': 0.03,
            'cache_ttl': 600,
            'confidence_threshold': 0.8,
            'max_workers': 8
        }
        engine = create_valuation_engine(config)
        self.assertEqual(engine.risk_free_rate, 0.03)
        self.assertEqual(engine.cache_ttl, 600)
        self.assertEqual(engine.confidence_threshold, 0.8)
        self.assertEqual(engine.max_workers, 8)
    
    def test_analyze_options_chain(self):
        """Test options chain analysis helper"""
        engine = create_valuation_engine()
        
        # Mock options data
        options_data = [
            {
                'strike': 150.0,
                'expiry': '2024-02-01',
                'type': 'call',
                'bid': 5.0,
                'ask': 5.5,
                'last': 5.25,
                'volume': 1000,
                'openInterest': 5000,
                'impliedVolatility': 0.25
            },
            {
                'strike': 155.0,
                'expiry': '2024-02-01',
                'type': 'put',
                'bid': 3.0,
                'ask': 3.5,
                'last': 3.25,
                'volume': 500,
                'openInterest': 2000,
                'impliedVolatility': 0.28
            }
        ]
        
        opportunities = analyze_options_chain(
            engine,
            symbol="AAPL",
            options_data=options_data,
            underlying_price=155.0
        )
        
        self.assertIsInstance(opportunities, list)


class TestValuationResult(unittest.TestCase):
    """Test ValuationResult data class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.5
        )
        
        self.valuation = ValuationResult(
            contract=self.contract,
            theoretical_value=6.0,
            market_price=5.25,
            mispricing=0.75,
            mispricing_ratio=0.14,
            confidence_score=80.0,
            pricing_breakdown={'black_scholes': 6.0, 'binomial': 5.9, 'monte_carlo': 6.1},
            greeks={'delta': 0.6, 'gamma': 0.02, 'theta': -0.05, 'vega': 0.15, 'rho': 0.08},
            implied_volatility=0.25,
            historical_volatility=0.22,
            iv_rank=45.0,
            iv_skew=0.02
        )
    
    def test_is_undervalued(self):
        """Test undervalued detection"""
        # Current valuation should be undervalued
        self.assertTrue(self.valuation.is_undervalued)
        
        # Create overvalued scenario
        overvalued = ValuationResult(
            contract=self.contract,
            theoretical_value=4.0,
            market_price=5.25,
            mispricing=-1.25,
            mispricing_ratio=-0.24,
            confidence_score=80.0,
            pricing_breakdown={},
            greeks={}
        )
        self.assertFalse(overvalued.is_undervalued)
    
    def test_opportunity_score(self):
        """Test opportunity score calculation"""
        score = self.valuation.opportunity_score
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        
        # Test with high-quality opportunity
        good_contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.5,
            volume=10000,
            open_interest=50000
        )
        
        good_valuation = ValuationResult(
            contract=good_contract,
            theoretical_value=7.0,
            market_price=5.25,
            mispricing=1.75,
            mispricing_ratio=0.33,
            confidence_score=90.0,
            pricing_breakdown={},
            greeks={'delta': 0.7}
        )
        
        good_score = good_valuation.opportunity_score
        self.assertGreater(good_score, 50)  # Should be a good opportunity


class TestOpportunityAnalysis(unittest.TestCase):
    """Test OpportunityAnalysis data class"""
    
    def test_opportunity_analysis_creation(self):
        """Test creating opportunity analysis"""
        contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.5
        )
        
        valuation = ValuationResult(
            contract=contract,
            theoretical_value=6.0,
            market_price=5.25,
            mispricing=0.75,
            mispricing_ratio=0.14,
            confidence_score=80.0,
            pricing_breakdown={},
            greeks={}
        )
        
        analysis = OpportunityAnalysis(
            valuation=valuation,
            expected_return=0.20,
            probability_of_profit=0.65,
            risk_reward_ratio=3.0,
            max_loss=5.25,
            max_gain=float('inf'),
            breakeven_price=155.25,
            suggested_position_size=0.05,
            confidence_level="high",
            key_risks=["High time decay risk"],
            entry_signals=["Undervalued by 14.3%"]
        )
        
        self.assertEqual(analysis.valuation, valuation)
        self.assertEqual(analysis.expected_return, 0.20)
        self.assertEqual(analysis.probability_of_profit, 0.65)
        self.assertEqual(analysis.confidence_level, "high")
        self.assertIn("High time decay risk", analysis.key_risks)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_valuation(self):
        """Test complete valuation workflow"""
        # Create engine
        engine = create_valuation_engine({
            'risk_free_rate': 0.04,
            'cache_ttl': 300
        })
        
        # Create option contract
        contract = OptionContract(
            symbol="SPY",
            strike=450.0,
            expiry=datetime.now() + timedelta(days=45),
            option_type=OptionType.CALL,
            style=OptionStyle.AMERICAN,
            bid=8.0,
            ask=8.5,
            last=8.25,
            volume=5000,
            open_interest=25000,
            underlying_price=455.0
        )
        
        # Create market data
        market_data = pd.DataFrame({
            'Close': np.random.normal(455, 10, 60)
        })
        
        # Perform valuation
        result = engine.detect_mispricing(contract, 455.0, market_data)
        
        # Verify result
        self.assertIsInstance(result, ValuationResult)
        self.assertIsNotNone(result.theoretical_value)
        self.assertIsNotNone(result.confidence_score)
        self.assertIn('delta', result.greeks)
        
        # Calculate expected returns
        expected_return, prob_profit = engine.calculate_expected_returns(
            result, target_price=460.0
        )
        
        self.assertIsInstance(expected_return, float)
        self.assertIsInstance(prob_profit, float)
    
    def test_batch_processing_performance(self):
        """Test batch processing with multiple options"""
        engine = create_valuation_engine({'max_workers': 4})
        
        # Create a chain of options
        expiry = datetime.now() + timedelta(days=30)
        options_chain = []
        
        for strike in range(145, 166, 5):
            for opt_type in [OptionType.CALL, OptionType.PUT]:
                contract = OptionContract(
                    symbol="AAPL",
                    strike=float(strike),
                    expiry=expiry,
                    option_type=opt_type,
                    bid=max(0.5, abs(155 - strike) * 0.8),
                    ask=max(0.5, abs(155 - strike) * 0.8 + 0.5),
                    volume=1000,
                    open_interest=5000,
                    underlying_price=155.0
                )
                options_chain.append(contract)
        
        # Scan opportunities
        import time
        start_time = time.time()
        
        opportunities = engine.scan_opportunities(
            options_chain,
            underlying_price=155.0,
            min_opportunity_score=0
        )
        
        elapsed_time = time.time() - start_time
        
        # Should process multiple options efficiently
        self.assertLess(elapsed_time, 10)  # Should complete within 10 seconds
        self.assertIsInstance(opportunities, list)
        
        # Verify opportunities are sorted by score
        if len(opportunities) > 1:
            scores = [opp.valuation.opportunity_score for opp in opportunities]
            self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_iv_surface_analysis(self):
        """Test IV surface analysis across strikes"""
        engine = create_valuation_engine()
        
        # Create options at different strikes
        expiry = datetime.now() + timedelta(days=30)
        options_chain = []
        
        for strike in [140, 145, 150, 155, 160, 165, 170]:
            contract = OptionContract(
                symbol="AAPL",
                strike=float(strike),
                expiry=expiry,
                option_type=OptionType.CALL,
                bid=max(0.5, 155 - strike + 2),
                ask=max(0.5, 155 - strike + 2.5),
                underlying_price=155.0
            )
            options_chain.append(contract)
        
        # Analyze IV surface
        surface_points = engine.analyze_iv_surface(
            options_chain,
            underlying_price=155.0
        )
        
        self.assertEqual(len(surface_points), 7)
        
        # Check moneyness calculation
        for point in surface_points:
            expected_moneyness = point.strike / 155.0
            self.assertAlmostEqual(point.moneyness, expected_moneyness, places=4)
        
        # IV should generally be positive
        for point in surface_points:
            self.assertGreater(point.implied_vol, 0)
    
    def test_caching_efficiency(self):
        """Test that caching improves performance"""
        engine = create_valuation_engine({'cache_ttl': 300})
        
        contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.5,
            underlying_price=155.0
        )
        
        import time
        
        # First calculation - no cache
        start_time = time.time()
        value1, _ = engine.calculate_fair_value(
            contract, 155.0, volatility=0.25, use_cache=True
        )
        first_calc_time = time.time() - start_time
        
        # Second calculation - should use cache
        start_time = time.time()
        value2, _ = engine.calculate_fair_value(
            contract, 155.0, volatility=0.25, use_cache=True
        )
        cached_calc_time = time.time() - start_time
        
        # Cached calculation should be much faster
        self.assertLess(cached_calc_time, first_calc_time * 0.1)
        self.assertEqual(value1, value2)
        
        # Verify cache stats
        stats = engine.get_cache_stats()
        self.assertGreater(stats['active_entries'], 0)
    
    @patch('data_feeds.options_valuation_engine.logger')
    def test_error_handling(self, mock_logger):
        """Test error handling in the engine"""
        engine = create_valuation_engine()
        
        # Test with invalid contract (no market price)
        bad_contract = OptionContract(
            symbol="AAPL",
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL
        )
        
        with self.assertRaises(ValueError):
            engine.detect_mispricing(bad_contract, 155.0)
        
        # Test with invalid options data in analyze_options_chain
        bad_options_data = [
            {'invalid': 'data'},
            {'strike': 'not_a_number', 'expiry': '2024-02-01', 'type': 'call'}
        ]
        
        opportunities = analyze_options_chain(
            engine,
            symbol="AAPL",
            options_data=bad_options_data,
            underlying_price=155.0
        )
        
        # Should handle errors gracefully
        self.assertIsInstance(opportunities, list)
        self.assertEqual(len(opportunities), 0)  # No valid opportunities from bad data


class TestModelValidation(unittest.TestCase):
    """Test pricing model validation against known values"""
    
    def test_black_scholes_put_call_parity(self):
        """Test put-call parity relationship"""
        engine = create_valuation_engine()
        
        S = 100  # Stock price
        K = 100  # Strike
        r = 0.05  # Risk-free rate
        T = 1.0  # Time to expiry
        sigma = 0.25  # Volatility
        
        # Calculate call and put prices
        call_price = engine._price_black_scholes(
            S, K, r, 0, sigma, T, OptionType.CALL
        )
        put_price = engine._price_black_scholes(
            S, K, r, 0, sigma, T, OptionType.PUT
        )
        
        # Put-call parity: C - P = S - K * exp(-r*T)
        parity_left = call_price - put_price
        parity_right = S - K * np.exp(-r * T)
        
        self.assertAlmostEqual(parity_left, parity_right, places=4)
    
    def test_option_price_bounds(self):
        """Test that option prices respect theoretical bounds"""
        engine = create_valuation_engine()
        
        S = 100
        K = 100
        r = 0.05
        T = 1.0
        sigma = 0.25
        
        # Call price bounds: max(S - K*exp(-rT), 0) <= C <= S
        call_price = engine._price_black_scholes(
            S, K, r, 0, sigma, T, OptionType.CALL
        )
        
        lower_bound = max(S - K * np.exp(-r * T), 0)
        upper_bound = S
        
        self.assertGreaterEqual(call_price, lower_bound)
        self.assertLessEqual(call_price, upper_bound)
        
        # Put price bounds: max(K*exp(-rT) - S, 0) <= P <= K*exp(-rT)
        put_price = engine._price_black_scholes(
            S, K, r, 0, sigma, T, OptionType.PUT
        )
        
        lower_bound = max(K * np.exp(-r * T) - S, 0)
        upper_bound = K * np.exp(-r * T)
        
        self.assertGreaterEqual(put_price, lower_bound)
        self.assertLessEqual(put_price, upper_bound)


if __name__ == '__main__':
    unittest.main()