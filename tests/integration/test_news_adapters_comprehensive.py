#!/usr/bin/env python3
"""
Comprehensive News Adapters Test Suite
Tests all available news adapters for functionality, data quality, and performance
"""

import logging
import sys
import os
import pytest
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import all news adapters
from data_feeds.news_adapters.base_news_adapter import BaseNewsAdapter
from data_feeds.news_adapters.reuters_adapter import ReutersAdapter
from data_feeds.news_adapters.benzinga_adapter import BenzingaAdapter
from data_feeds.news_adapters.cnn_business_adapter import CNNBusinessAdapter
from data_feeds.news_adapters.financial_times_adapter import FinancialTimesAdapter
from data_feeds.news_adapters.fortune_adapter import FortuneAdapter
from data_feeds.news_adapters.marketwatch_adapter import MarketWatchAdapter
from data_feeds.news_adapters.seeking_alpha_adapter import SeekingAlphaAdapter

logger = logging.getLogger(__name__)

@dataclass
class NewsAdapterTestResult:
    """Test result for a news adapter"""
    adapter_name: str
    success: bool
    articles_count: int
    response_time: float
    error_message: Optional[str] = None
    sample_article: Optional[Dict[str, Any]] = None
    sentiment_available: bool = False
    quality_score: float = 0.0

class NewsAdapterTester:
    """Comprehensive news adapter testing framework"""
    
    def __init__(self):
        self.test_symbols = ['AAPL', 'TSLA', 'MSFT', 'NVDA']
        self.adapters = {
            'reuters': ReutersAdapter(),
            'benzinga': BenzingaAdapter(),
            'cnn_business': CNNBusinessAdapter(),
            'financial_times': FinancialTimesAdapter(),
            'fortune': FortuneAdapter(),
            'marketwatch': MarketWatchAdapter(),
            'seeking_alpha': SeekingAlphaAdapter(),
        }
        
    def test_single_adapter(self, adapter_name: str, adapter: BaseNewsAdapter, 
                          symbol: str, limit: int = 10) -> NewsAdapterTestResult:
        """Test a single news adapter"""
        start_time = time.time()
        
        try:
            logger.info(f"Testing {adapter_name} adapter for {symbol}")
            
            # Fetch articles
            articles = adapter.fetch_news_articles(symbol, limit=limit)
            response_time = time.time() - start_time
            
            if not articles:
                return NewsAdapterTestResult(
                    adapter_name=adapter_name,
                    success=False,
                    articles_count=0,
                    response_time=response_time,
                    error_message="No articles returned"
                )
            
            # Validate article structure
            sample_article = articles[0] if articles else None
            quality_score = self._calculate_quality_score(articles, adapter_name)
            
            # Check for sentiment capability
            sentiment_available = hasattr(adapter, 'sentiment_engine') and adapter.sentiment_engine is not None
            
            return NewsAdapterTestResult(
                adapter_name=adapter_name,
                success=True,
                articles_count=len(articles),
                response_time=response_time,
                sample_article=sample_article,
                sentiment_available=sentiment_available,
                quality_score=quality_score
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error testing {adapter_name}: {e}")
            
            return NewsAdapterTestResult(
                adapter_name=adapter_name,
                success=False,
                articles_count=0,
                response_time=response_time,
                error_message=str(e)
            )
    
    def _calculate_quality_score(self, articles: List[Dict[str, Any]], adapter_name: str) -> float:
        """Calculate quality score for articles"""
        if not articles:
            return 0.0
        
        score = 100.0
        issues = []
        
        for article in articles:
            # Check required fields
            if not article.get('title'):
                issues.append("Missing title")
                score -= 10
            
            if not article.get('description'):
                issues.append("Missing description")
                score -= 5
                
            if not article.get('link'):
                issues.append("Missing link")
                score -= 5
                
            if not article.get('published'):
                issues.append("Missing published date")
                score -= 5
        
        # Penalize for too few articles
        if len(articles) < 3:
            score -= (3 - len(articles)) * 10
            
        return max(0.0, score / len(articles))
    
    def test_all_adapters(self, symbol: str = 'AAPL') -> List[NewsAdapterTestResult]:
        """Test all news adapters for a given symbol"""
        results = []
        
        for adapter_name, adapter in self.adapters.items():
            result = self.test_single_adapter(adapter_name, adapter, symbol)
            results.append(result)
            
            # Brief pause between tests to avoid rate limiting
            time.sleep(1)
        
        return results
    
    def generate_test_report(self, results: List[NewsAdapterTestResult]) -> str:
        """Generate a comprehensive test report"""
        report_lines = [
            "=" * 80,
            "NEWS ADAPTERS COMPREHENSIVE TEST REPORT",
            "=" * 80,
            f"Test Date: {datetime.now().isoformat()}",
            f"Total Adapters Tested: {len(results)}",
            ""
        ]
        
        # Summary statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        report_lines.extend([
            "SUMMARY:",
            f"  ✅ Successful: {len(successful)}/{len(results)}",
            f"  ❌ Failed: {len(failed)}/{len(results)}",
            f"  Average Response Time: {sum(r.response_time for r in results) / len(results):.2f}s",
            f"  Average Quality Score: {sum(r.quality_score for r in successful) / len(successful) if successful else 0:.1f}",
            ""
        ])
        
        # Detailed results
        report_lines.append("DETAILED RESULTS:")
        for result in results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            report_lines.extend([
                f"\n{result.adapter_name.upper()} - {status}",
                f"  Articles Count: {result.articles_count}",
                f"  Response Time: {result.response_time:.2f}s",
                f"  Quality Score: {result.quality_score:.1f}",
                f"  Sentiment Available: {'Yes' if result.sentiment_available else 'No'}"
            ])
            
            if result.error_message:
                report_lines.append(f"  Error: {result.error_message}")
            
            if result.sample_article:
                article = result.sample_article
                report_lines.extend([
                    f"  Sample Article:",
                    f"    Title: {article.get('title', 'N/A')[:100]}...",
                    f"    Source: {article.get('source', 'N/A')}",
                    f"    Published: {article.get('published', 'N/A')}"
                ])
        
        # Recommendations
        report_lines.extend([
            "\n" + "=" * 80,
            "RECOMMENDATIONS:",
            ""
        ])
        
        if failed:
            report_lines.append("Failed Adapters:")
            for result in failed:
                report_lines.append(f"  - {result.adapter_name}: {result.error_message}")
            report_lines.append("")
        
        if successful:
            # Sort by quality score
            successful.sort(key=lambda x: x.quality_score, reverse=True)
            report_lines.append("Top Performing Adapters:")
            for result in successful[:3]:
                report_lines.append(f"  - {result.adapter_name}: {result.quality_score:.1f} quality, {result.response_time:.2f}s")
        
        return "\n".join(report_lines)

def test_reuters_adapter():
    """Test Reuters adapter specifically"""
    tester = NewsAdapterTester()
    result = tester.test_single_adapter('reuters', tester.adapters['reuters'], 'AAPL')
    
    assert result.adapter_name == 'reuters'
    if result.success:
        assert result.articles_count > 0
        assert result.response_time > 0
        assert result.sample_article is not None
        
def test_all_news_adapters():
    """Test all news adapters"""
    tester = NewsAdapterTester()
    results = tester.test_all_adapters('AAPL')
    
    assert len(results) == len(tester.adapters)
    
    # At least one adapter should work
    successful = [r for r in results if r.success]
    assert len(successful) > 0, "No news adapters are working"

def main():
    """Main test execution"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create tester
    tester = NewsAdapterTester()
    
    print("🚀 Starting Comprehensive News Adapters Test Suite...")
    print(f"Testing {len(tester.adapters)} adapters...")
    
    # Test all adapters
    results = tester.test_all_adapters('AAPL')
    
    # Generate and print report
    report = tester.generate_test_report(results)
    print(report)
    
    # Save report to file
    with open('news_adapters_test_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n📊 Test report saved to: news_adapters_test_report.txt")
    
    # Return success if at least 50% of adapters work
    successful = [r for r in results if r.success]
    success_rate = len(successful) / len(results)
    
    if success_rate >= 0.5:
        print(f"✅ Test Suite PASSED: {len(successful)}/{len(results)} adapters working")
        return 0
    else:
        print(f"❌ Test Suite FAILED: Only {len(successful)}/{len(results)} adapters working")
        return 1

if __name__ == "__main__":
    sys.exit(main())
