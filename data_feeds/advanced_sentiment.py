"""
Advanced Sentiment Analysis Engine
Multi-model ensemble with VADER, FinBERT, and custom financial lexicon
Provides confidence-weighted sentiment scoring for trading decisions
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TextBlob = None
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Sentiment will use VADER and FinBERT only.")

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Enhanced sentiment result with multi-model scoring"""
    symbol: str
    text_snippet: str
    vader_score: float
    finbert_score: float
    ensemble_score: float
    confidence: float
    timestamp: datetime
    model_weights: Dict[str, float]
    text_length: int
    source: str

@dataclass
class SentimentSummary:
    """Aggregated sentiment summary for a symbol"""
    symbol: str
    overall_sentiment: float
    confidence: float
    sample_size: int
    bullish_mentions: int
    bearish_mentions: int
    neutral_mentions: int
    trending_direction: str  # "bullish", "bearish", "neutral", "uncertain"
    timestamp: datetime
    quality_score: float

class FinancialLexicon:
    """Enhanced financial sentiment lexicon with sector/macro/context markers and dispersion-aware scoring."""
    
    def __init__(self):
        # Bullish terms with intensity scores
        self.bullish_terms = {
            'moon': 2.0, 'rocket': 1.8, 'bull': 1.5, 'buy': 1.2, 'long': 1.1,
            'breakout': 1.6, 'uptrend': 1.4, 'rally': 1.3, 'surge': 1.5,
            'outperform': 1.2, 'upgrade': 1.3, 'target': 1.1, 'beat': 1.4,
            'strong': 1.2, 'momentum': 1.1, 'bullish': 1.5, 'pump': 1.8,
            'hodl': 1.3, 'diamond hands': 2.0, 'to the moon': 2.5,
            'earnings beat': 1.6, 'guidance raise': 1.7, 'revenue growth': 1.3,
            'pricing power': 1.4, 'backlog expanded': 1.2, 'book-to-bill': 1.1,
            'beat and raise': 1.8, 'reacceleration': 1.3, 'visibility improved': 1.1
        }
        
        # Bearish terms with intensity scores
        self.bearish_terms = {
            'crash': -2.0, 'dump': -1.8, 'bear': -1.5, 'sell': -1.2, 'short': -1.1,
            'breakdown': -1.6, 'downtrend': -1.4, 'decline': -1.3, 'plunge': -1.5,
            'underperform': -1.2, 'downgrade': -1.3, 'miss': -1.4, 'weak': -1.2,
            'bearish': -1.5, 'bubble': -1.6, 'overvalued': -1.3, 'resistance': -1.1,
            'paper hands': -1.8, 'rugpull': -2.5, 'bag holder': -1.9,
            'earnings miss': -1.6, 'guidance cut': -1.7, 'revenue decline': -1.3,
            'margin compression': -1.4, 'inventory overhang': -1.2, 'order cancellations': -1.3,
            'probe': -1.1, 'settlement': -0.6, 'class action': -1.2, 'regulatory risk': -1.1,
            'weak sell-through': -1.0, 'credit losses': -1.1, 'liquidity stress': -1.2
        }
        
        # Macro/sector regime terms (contextual, smaller weights)
        self.regime_terms = {
            'volatility': -0.5, 'uncertainty': -0.8, 'correction': -1.2,
            'rotation': 0.0, 'consolidation': -0.3, 'institutional': 0.5,
            'retail': 0.2, 'volume': 0.1, 'options flow': 0.3,
            'soft landing': 0.6, 'hard landing': -0.9, 'disinflation': 0.5,
            'stagflation': -1.4, 'tightening': -0.6, 'easing': 0.5,
            'recession risk': -1.1, 'credit crunch': -1.2
        }

        # Qualifiers: intensifiers/diminishers/negations affect weighting
        self.intensifiers = {'materially': 1.25, 'significantly': 1.2, 'substantially': 1.2, 'meaningfully': 1.15, 'strongly': 1.1}
        self.diminishers = {'slightly': 0.85, 'modestly': 0.9, 'marginally': 0.9, 'somewhat': 0.92}
        self.negations = {"not", "no", "without", "lack", "isn't", "wasn't", "aren't", "weren't", "never"}
        
        # Forward-looking markers vs backward-looking
        self.forward_markers = {'expects', 'guiding', 'outlook', 'forecast', 'project', 'anticipate', 'targets', 'sees'}
        self.backward_markers = {'reported', 'was', 'were', 'had', 'realized'}

    def _apply_qualifiers(self, text_tokens: List[str], base_weight: float, hit_index: int) -> float:
        window = 3
        w = base_weight
        start = max(0, hit_index - window)
        end = min(len(text_tokens), hit_index + window + 1)
        context = set(text_tokens[start:end])
        if any(tok in self.negations for tok in context):
            w = -abs(w) * 0.8 if w > 0 else abs(w) * 0.8
        for tok in context:
            if tok in self.intensifiers:
                w *= self.intensifiers[tok]
            elif tok in self.diminishers:
                w *= self.diminishers[tok]
        return w

    def _context_boost(self, text_tokens: List[str], score: float) -> float:
        tokens = set(text_tokens)
        if any(t in tokens for t in self.forward_markers):
            score *= 1.08
        if any(t in tokens for t in self.backward_markers):
            score *= 0.96
        return score
    
    def get_lexicon_score(self, text: str) -> Tuple[float, int]:
        """Get sentiment score based on financial lexicon with contextual qualifiers"""
        text_lower = text.lower()
        tokens = text_lower.split()
        score = 0.0
        matches = 0
        for terms_dict in [self.bullish_terms, self.bearish_terms, self.regime_terms]:
            for term, weight in terms_dict.items():
                if term in text_lower:
                    try:
                        term_first = term.split()[0]
                        hit_index = tokens.index(term_first) if term_first in tokens else 0
                    except Exception:
                        hit_index = 0
                    adj_weight = self._apply_qualifiers(tokens, weight, hit_index)
                    score += adj_weight
                    matches += 1
        score = self._context_boost(tokens, score)
        return score, matches

class FinBERTAnalyzer:
    """FinBERT-based financial sentiment analyzer"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        self._load_lock = threading.Lock()
        # Single re-entrant lock to serialize tokenizer/pipeline usage and avoid "Already borrowed"
        self._instance_lock = threading.RLock()
        # Dedicated inference lock to prevent concurrent HF tokenizer borrow issues
        self._inference_lock = threading.Lock()
        
        # Try to load model on initialization (non-blocking)
        self._load_model()
    
    def _load_model(self):
        """Load FinBERT model with error handling"""
        try:
            with self._load_lock:
                if self.is_loaded:
                    return
                
                logger.info(f"Loading FinBERT model: {self.model_name}")
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.eval()
                
                # Create pipeline for easier inference (disable parallelism to avoid HF tokenizers borrow issues)
                import os
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                self.is_loaded = True
                logger.info("FinBERT model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.is_loaded = False
    
    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment with FinBERT
        Returns (sentiment_score, confidence)
        """
        if not self.is_loaded:
            # Try to load model again
            self._load_model()
        if not self.is_loaded or self.tokenizer is None or self.pipeline is None:
            return 0.0, 0.0

        try:
            # To fully avoid shared Encoding borrows, bypass pipeline and call model directly with fresh tensors
            max_length = 512
            with self._inference_lock:
                enc = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    padding=False
                )
                input_ids = enc.get("input_ids")
                attention_mask = enc.get("attention_mask")
                if input_ids is None or input_ids.numel() == 0:
                    return 0.0, 0.0

            # Inference: call model directly (no pipeline) to avoid any internal parallel tokenizers usage
            with self._instance_lock, torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                # Softmax to get probabilities
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                # FinBERT label order typically: [negative, neutral, positive]
                neg, neu, pos = float(probs[0]), float(probs[1]), float(probs[2])
                # Choose label with highest probability
                if pos >= max(neg, neu):
                    label = "POSITIVE"
                    score_val = pos
                elif neg >= max(pos, neu):
                    label = "NEGATIVE"
                    score_val = neg
                else:
                    label = "NEUTRAL"
                    score_val = neu

                # Build result dict compatible with previous logic
                result = {"label": label, "score": score_val}

            # Convert to standardized score (-1 to 1)
            label = result.get('label', '').upper()
            confidence = float(result.get('score', 0.0) or 0.0)

            if label == 'POSITIVE':
                sentiment_score = confidence
            elif label == 'NEGATIVE':
                sentiment_score = -confidence
            else:  # NEUTRAL or unknown
                sentiment_score = 0.0

            return float(sentiment_score), confidence

        except Exception as e:
            # Log once without duplicating upstream
            logger.error(f"FinBERT analysis error: {e}")
            return 0.0, 0.0

class AdvancedSentimentEngine:
    """
    Multi-model sentiment analysis engine for financial markets
    Combines VADER, FinBERT, and custom financial lexicon
    """
    
    def __init__(self):
        # Initialize analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.finbert_analyzer = FinBERTAnalyzer()
        self.financial_lexicon = FinancialLexicon()
        self.textblob_available = TEXTBLOB_AVAILABLE
        
        # Model performance tracking
        self.model_performance = {
            'vader': {'accuracy': 0.7, 'weight': 0.3},
            'finbert': {'accuracy': 0.85, 'weight': 0.5},
            'lexicon': {'accuracy': 0.6, 'weight': 0.2}
        }
        
        # Cache for processed sentiment
        self.cache = {}
        self.cache_duration = timedelta(minutes=30)
        
        # Threading for parallel processing
        self.max_workers = 4
        
        logger.info("AdvancedSentimentEngine initialized")
    
    def analyze_text(self, text: str, symbol: str, source: str = "unknown") -> SentimentResult:
        """
        Analyze single text with all models, robust fallback for model errors
        """
        start_time = time.time()

        # VADER analysis
        try:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_sentiment = vader_scores['compound']
        except Exception as e:
            logger.error(f"VADER analysis error: {e}")
            vader_scores = {'compound': 0.0}
            vader_sentiment = 0.0

        # TextBlob analysis (optional, fallback)
        textblob_sentiment = None
        if self.textblob_available and TextBlob is not None:
            try:
                blob = TextBlob(text)
                # TextBlob's sentiment property is a namedtuple with .polarity
                textblob_sentiment = getattr(getattr(blob, 'sentiment', None), 'polarity', None)
            except Exception as e:
                logger.warning(f"TextBlob sentiment failed: {e}")
                textblob_sentiment = None

        # Financial lexicon analysis
        try:
            lexicon_score, lexicon_matches = self.financial_lexicon.get_lexicon_score(text)
            lexicon_sentiment = np.tanh(lexicon_score / 3.0)  # Normalize to [-1, 1]
        except Exception as e:
            logger.error(f"Lexicon analysis error: {e}")
            lexicon_score, lexicon_matches, lexicon_sentiment = 0.0, 0, 0.0

        # FinBERT analysis (guarded)
        try:
            finbert_sentiment, finbert_confidence = self.finbert_analyzer.analyze(text)
        except Exception as e:
            # Already logged inside analyzer; avoid duplicate logs here
            finbert_sentiment, finbert_confidence = 0.0, 0.0

        # Calculate ensemble score with dynamic weighting
        weights = self._get_dynamic_weights(len(text), lexicon_matches, finbert_confidence)

        # If all models fail, fallback to 0.0, else use available scores
        model_scores = [vader_sentiment, lexicon_sentiment, finbert_sentiment]
        if all(abs(s) < 1e-6 for s in model_scores):
            # All models failed, fallback to TextBlob if available
            if textblob_sentiment is not None:
                ensemble_score = textblob_sentiment
            else:
                ensemble_score = 0.0
        else:
            ensemble_score = (
                vader_sentiment * weights['vader'] +
                lexicon_sentiment * weights['lexicon'] +
                finbert_sentiment * weights['finbert']
            )
            # Optionally blend in TextBlob if available (not weighted in original ensemble)
            if textblob_sentiment is not None:
                ensemble_score = (ensemble_score + textblob_sentiment) / 2

        # If ensemble_score is still 0, but at least one model had confidence, use the highest-confidence model
        if abs(ensemble_score) < 1e-6:
            candidates = [
                (abs(vader_sentiment), vader_sentiment),
                (abs(lexicon_sentiment), lexicon_sentiment),
                (abs(finbert_sentiment), finbert_sentiment)
            ]
            best = max(candidates, key=lambda x: x[0])
            if best[0] > 0:
                ensemble_score = best[1]

        # Calculate overall confidence with simple dispersion penalty for inter-model disagreement
        try:
            model_vals = [vader_sentiment, lexicon_sentiment, finbert_sentiment]
            dispersion = np.std(model_vals) if len(model_vals) >= 2 else 0.0
            dispersion_penalty = float(min(0.15, dispersion * 0.2))  # cap penalty
        except Exception:
            dispersion_penalty = 0.0

        confidence = max(0.0, self._calculate_confidence(
            vader_scores, lexicon_matches, finbert_confidence, len(text)
        ) - dispersion_penalty)

        result = SentimentResult(
            symbol=symbol,
            text_snippet=text[:100] + "..." if len(text) > 100 else text,
            vader_score=vader_sentiment,
            finbert_score=finbert_sentiment,
            ensemble_score=ensemble_score,
            confidence=confidence,
            timestamp=datetime.now(),
            model_weights=weights,
            text_length=len(text),
            source=source
        )

        processing_time = time.time() - start_time
        logger.debug(f"Sentiment analysis completed in {processing_time:.2f}s for {symbol}")

        return result
    
    def analyze_batch(self, texts: Optional[List[str]], symbols: Optional[List[str]], sources: Optional[List[str]] = None) -> List[SentimentResult]:
        """
        Analyze multiple texts in parallel
        """
        if texts is None:
            texts = []
        if symbols is None:
            symbols = []
        if sources is None:
            sources = ["unknown"] * len(texts)

        # Defensive: ensure all are lists and same length
        if not isinstance(texts, list):
            texts = list(texts)
        if not isinstance(symbols, list):
            symbols = list(symbols)
        if not isinstance(sources, list):
            sources = list(sources)

        if len(texts) != len(symbols) or len(texts) != len(sources):
            raise ValueError("All input lists must have the same length")

        results = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(self.analyze_text, text, symbol, source): (text, symbol, source)
                for text, symbol, source in zip(texts, symbols, sources)
            }

            # Collect results as they complete
            for future in as_completed(future_to_args):
                try:
                    result = future.result(timeout=30)  # 30 second timeout per text
                    results.append(result)
                except Exception as e:
                    text, symbol, source = future_to_args[future]
                    logger.error(f"Failed to analyze sentiment for {symbol}: {e}")

        return results
    
    def get_symbol_sentiment_summary(self, symbol: str, texts: Optional[List[str]], sources: Optional[List[str]] = None) -> SentimentSummary:
        """
        Get aggregated sentiment summary for a symbol
        """
        if texts is None:
            texts = []
        if not texts:
            return self._empty_sentiment_summary(symbol)

        # Defensive: ensure texts is a list
        if not isinstance(texts, list):
            texts = list(texts)

        # Analyze all texts
        symbols = [symbol] * len(texts)
        results = self.analyze_batch(texts, symbols, sources)

        if not results:
            return self._empty_sentiment_summary(symbol)

        # Aggregate results
        ensemble_scores = [r.ensemble_score for r in results]
        confidences = [r.confidence for r in results]

        # Weight sentiment by confidence
        weighted_sentiment = sum(score * conf for score, conf in zip(ensemble_scores, confidences))
        total_confidence = sum(confidences)

        if total_confidence > 0:
            overall_sentiment = weighted_sentiment / total_confidence
            avg_confidence = total_confidence / len(results)
        else:
            overall_sentiment = 0.0
            avg_confidence = 0.0

        # Count sentiment directions
        bullish = sum(1 for score in ensemble_scores if score > 0.1)
        bearish = sum(1 for score in ensemble_scores if score < -0.1)
        neutral = len(ensemble_scores) - bullish - bearish

        # Determine trending direction
        trending_direction = self._determine_trend(overall_sentiment, avg_confidence, bullish, bearish, neutral)

        # Calculate quality score
        quality_score = min(100, avg_confidence * 100 + len(results) * 2)

        return SentimentSummary(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            confidence=avg_confidence,
            sample_size=len(results),
            bullish_mentions=bullish,
            bearish_mentions=bearish,
            neutral_mentions=neutral,
            trending_direction=trending_direction,
            timestamp=datetime.now(),
            quality_score=quality_score
        )
    
    def _get_dynamic_weights(self, text_length: int, lexicon_matches: int, finbert_confidence: float) -> Dict[str, float]:
        """Calculate dynamic weights based on context"""
        # Base weights
        weights = self.model_performance.copy()
        
        # Adjust based on text quality
        if text_length < 50:  # Short text - favor lexicon
            weights['lexicon']['weight'] *= 1.5
            weights['finbert']['weight'] *= 0.7
        elif text_length > 200:  # Long text - favor FinBERT
            weights['finbert']['weight'] *= 1.3
            weights['vader']['weight'] *= 0.8
        
        # Adjust based on lexicon matches
        if lexicon_matches > 3:  # Many financial terms - trust lexicon more
            weights['lexicon']['weight'] *= 1.4
        
        # Adjust based on FinBERT confidence
        if finbert_confidence > 0.8:  # High FinBERT confidence
            weights['finbert']['weight'] *= 1.2
        elif finbert_confidence < 0.5:  # Low FinBERT confidence
            weights['finbert']['weight'] *= 0.6
        
        # Normalize weights
        total_weight = sum(w['weight'] for w in weights.values())
        return {model: w['weight'] / total_weight for model, w in weights.items()}
    
    def _calculate_confidence(self, vader_scores: Dict, lexicon_matches: int, finbert_confidence: float, text_length: int) -> float:
        """Calculate overall confidence score"""
        # Base confidence from VADER
        vader_conf = abs(vader_scores['compound'])
        
        # Lexicon confidence based on matches
        lexicon_conf = min(1.0, lexicon_matches / 5.0)
        
        # Text length factor
        length_factor = min(1.0, text_length / 100.0)
        
        # Combine confidences
        overall_confidence = (
            vader_conf * 0.3 +
            lexicon_conf * 0.2 +
            finbert_confidence * 0.4 +
            length_factor * 0.1
        )
        
        return min(1.0, overall_confidence)
    
    def _determine_trend(self, sentiment: float, confidence: float, bullish: int, bearish: int, neutral: int) -> str:
        """Determine trending direction"""
        if confidence < 0.3:
            return "uncertain"
        
        if sentiment > 0.2 and bullish > bearish:
            return "bullish"
        elif sentiment < -0.2 and bearish > bullish:
            return "bearish"
        else:
            return "neutral"
    
    def _empty_sentiment_summary(self, symbol: str) -> SentimentSummary:
        """Return empty sentiment summary"""
        return SentimentSummary(
            symbol=symbol,
            overall_sentiment=0.0,
            confidence=0.0,
            sample_size=0,
            bullish_mentions=0,
            bearish_mentions=0,
            neutral_mentions=0,
            trending_direction="uncertain",
            timestamp=datetime.now(),
            quality_score=0.0
        )
    
    def update_model_performance(self, model_name: str, accuracy: float):
        """Update model performance tracking for dynamic weighting"""
        if model_name in self.model_performance:
            self.model_performance[model_name]['accuracy'] = accuracy
            logger.info(f"Updated {model_name} accuracy to {accuracy:.2f}")

# ============================================================================
# Public Interface Functions
# ============================================================================

# Global engine instance
_sentiment_engine = None

def get_sentiment_engine() -> AdvancedSentimentEngine:
    """Get global sentiment engine instance"""
    global _sentiment_engine
    if _sentiment_engine is None:
        _sentiment_engine = AdvancedSentimentEngine()
    return _sentiment_engine

def analyze_symbol_sentiment(symbol: str, texts: Optional[List[str]], sources: Optional[List[str]] = None) -> SentimentSummary:
    """Analyze sentiment for a symbol using all available texts"""
    if texts is None:
        texts = []
    if not isinstance(texts, list):
        texts = list(texts)
    return get_sentiment_engine().get_symbol_sentiment_summary(symbol, texts, sources)

def analyze_text_sentiment(text: str, symbol: str, source: str = "unknown") -> SentimentResult:
    """Analyze sentiment for a single text"""
    return get_sentiment_engine().analyze_text(text, symbol, source)

def batch_analyze_sentiment(texts: Optional[List[str]], symbols: Optional[List[str]], sources: Optional[List[str]] = None) -> List[SentimentResult]:
    """Analyze sentiment for multiple texts in parallel"""
    if texts is None:
        texts = []
    if symbols is None:
        symbols = []
    if not isinstance(texts, list):
        texts = list(texts)
    if not isinstance(symbols, list):
        symbols = list(symbols)
    return get_sentiment_engine().analyze_batch(texts, symbols, sources)
