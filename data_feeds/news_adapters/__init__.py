"""
News Adapters Module
Enhanced financial news sources with advanced sentiment analysis
"""

from .base_news_adapter import BaseNewsAdapter
from .reuters_adapter import ReutersAdapter
from .marketwatch_adapter import MarketWatchAdapter
from .cnn_business_adapter import CNNBusinessAdapter
from .financial_times_adapter import FinancialTimesAdapter

__all__ = [
    'BaseNewsAdapter',
    'ReutersAdapter', 
    'MarketWatchAdapter',
    'CNNBusinessAdapter',
    'FinancialTimesAdapter'
]
