"""
Advanced Sentiment Analysis Engine
Multi-model ensemble with VADER, FinBERT, and custom financial lexicon
Provides confidence-weighted sentiment scoring for trading decisions
"""

import logging
import time
import os  # Needed for ADV_SENTIMENT_VERBOSE flag and finbert verbosity gating
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Suppress FutureWarning from transformers library about encoder_attention_mask
warnings.filterwarnings("ignore", message=r".*encoder_attention_mask.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Do not create a HF pipeline here; we call the model/tokenizer directly to avoid tokenizer parallelism issues
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TextBlob = None
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Sentiment will use VADER and FinBERT only.")

# Add HTTP error handling for rate limiting
try:
    from requests.exceptions import HTTPError
except ImportError:
    HTTPError = Exception  # Fallback if requests not available

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
    processing_time: float = 0.0
    sample_size: Optional[int] = None

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

# Global sentiment engine instance
_sentiment_engine = None

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
    """Enhanced FinBERT-based financial sentiment analyzer with improved model selection and robustness"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        self._load_lock = threading.Lock()
        self._instance_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        
        # Enhanced model options for better accuracy - prioritize stable, reliable models
        self.model_options = {
            'finbert_news': "ahmedrachid/FinancialBERT-Sentiment-Analysis",  # News-focused sentiment (stable)
            'finbert_earnings': "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",  # Earnings-specific (stable)
            'finbert_prosus': "ProsusAI/finbert",  # Original as fallback (stable)
            'fintwitbert_sentiment': "StephanAkkerman/FinTwitBERT-sentiment",  # Twitter-specialized financial sentiment (stable)
        }
        
        # Model performance tracking for dynamic selection (updated for stable models only)
        self.model_performance = {
            'finbert_news': {'accuracy': 0.87, 'latency': 1.3, 'financial_relevance': 0.94},
            'finbert_earnings': {'accuracy': 0.89, 'latency': 1.4, 'financial_relevance': 0.96},
            'finbert_prosus': {'accuracy': 0.82, 'latency': 1.0, 'financial_relevance': 0.88},
            'fintwitbert_sentiment': {'accuracy': 0.91, 'latency': 1.3, 'financial_relevance': 0.98},  # High relevance for Twitter
        }
        
        # Try to load model on initialization (non-blocking)
        self._load_model()
    
    def _select_model_for_source(self, source: str) -> str:
        """Select the best model based on the content source, using available stable models"""
        source_lower = source.lower()
        
        # Use FinTwitBERT for Twitter and Reddit (specialized for social media financial sentiment)
        if 'twitter' in source_lower or 'tweet' in source_lower or 'social' in source_lower:
            return 'fintwitbert_sentiment'
        elif 'reddit' in source_lower:
            return 'fintwitbert_sentiment'
        else:
            # For news and other sources, use finbert_news (better suited for structured content)
            return 'finbert_news'
    
    def _ensure_correct_model(self, source: str):
        """Ensure the correct model is loaded for the given source"""
        # Skip model switching if source is empty (used during accuracy testing)
        if not source or source.strip() == "":
            return

        preferred_model = self._select_model_for_source(source)
        if self.model_name != self.model_options[preferred_model]:
            logger.info(f"Switching model from {self.model_name} to {self.model_options[preferred_model]} for source: {source}")
            self.model_name = self.model_options[preferred_model]
            self.is_loaded = False
            self._load_model()
    
    def _load_model_with_retry(self, model_name: str, max_retries: int = 5) -> Tuple[Optional[Any], Optional[Any]]:
        """Load model with enhanced retry logic for rate limiting"""
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Loading model {model_name} (attempt {attempt + 1}/{max_retries + 1})")
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model.eval()
                return tokenizer, model
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = '429' in error_msg or 'too many requests' in error_msg or 'rate limit' in error_msg
                
                if is_rate_limit and attempt < max_retries:
                    # Exponential backoff for rate limiting: 2^attempt seconds, max 60 seconds
                    backoff_time = min(2 ** attempt, 60)
                    logger.warning(f"Rate limit hit for {model_name}, retrying in {backoff_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(backoff_time)
                    continue
                elif attempt < max_retries:
                    # For other errors, shorter backoff
                    backoff_time = min(1 * (attempt + 1), 10)
                    logger.warning(f"Model load failed for {model_name}: {e}, retrying in {backoff_time}s")
                    time.sleep(backoff_time)
                    continue
                else:
                    logger.error(f"Failed to load model {model_name} after {max_retries + 1} attempts: {e}")
                    return None, None
        
        return None, None
    
    def _load_model(self):
        """Load FinBERT model with enhanced error handling and intelligent model selection"""
        try:
            load_start = time.time()
            with self._load_lock:
                if self.is_loaded:
                    return
                
                logger.info(f"Loading FinBERT model: {self.model_name}")
                
                # Try primary model first with retry logic
                self.tokenizer, self.model = self._load_model_with_retry(self.model_name)
                
                if self.tokenizer is None or self.model is None:
                    logger.warning(f"Primary model {self.model_name} failed to load, trying fallbacks")
                    
                    # Try fallback models in order of performance
                    performance_ranked = sorted(
                        self.model_options.items(),
                        key=lambda x: self.model_performance.get(x[0], {}).get('accuracy', 0),
                        reverse=True
                    )
                    
                    for model_key, fallback_model in performance_ranked:
                        if fallback_model != self.model_name:
                            logger.info(f"Trying fallback model: {fallback_model}")
                            self.tokenizer, self.model = self._load_model_with_retry(fallback_model)
                            if self.tokenizer is not None and self.model is not None:
                                self.model_name = fallback_model
                                logger.info(f"Successfully loaded fallback model: {fallback_model}")
                                break
                            else:
                                logger.warning(f"Fallback model {fallback_model} also failed")
                                continue
                    else:
                        raise RuntimeError("All FinBERT models failed to load")
                
                # Enhanced tokenizer configuration for better financial text processing
                import os
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                self.pipeline = None
                
                self.is_loaded = True
                logger.info(f"FinBERT model loaded successfully: {self.model_name}")
                load_elapsed = time.time() - load_start
                logger.info(f"[TIMING][advanced_sentiment][finbert_load] {load_elapsed:.2f} seconds to load FinBERT")
                
        except Exception as e:
            logger.error(f"Failed to load any FinBERT model: {e}")
            self.is_loaded = False
    
    def analyze(self, text: str, source: str = "unknown") -> Tuple[float, float]:
        """
        Analyze sentiment with FinBERT with enhanced financial context awareness
        Selects appropriate model based on source (e.g., FinTwitBERT for Twitter)
        Returns (sentiment_score, confidence)
        """
        # Ensure correct model is loaded for the source
        self._ensure_correct_model(source)

        retries = 0
        max_retries = int(os.getenv("FINBERT_MAX_RETRIES", "2"))
        while True:
            try:
                # Enhanced text preprocessing for financial content
                processed_text = self._preprocess_financial_text(text)
                
                # To fully avoid shared Encoding borrows, bypass pipeline and call model directly with fresh tensors
                max_length = 512
                with self._inference_lock:
                    enc_start = time.time()
                    enc = self.tokenizer(
                        processed_text,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                        padding=False
                    )
                    enc_elapsed = time.time() - enc_start
                    if os.getenv("ADV_FINBERT_VERBOSE", "0") == "1":
                        logger.info(f"[TIMING][advanced_sentiment][finbert_tokenize] {enc_elapsed:.2f} seconds")
                    input_ids = enc.get("input_ids")
                    attention_mask = enc.get("attention_mask")
                    if input_ids is None or input_ids.numel() == 0:
                        return 0.0, 0.0

                # Inference: call model directly (no pipeline) to avoid any internal parallel tokenizers usage
                with self._instance_lock, torch.no_grad():
                    model_start = time.time()
                    # self.model should be loaded and not None; assert for static check
                    assert self.model is not None, "FinBERT model is not loaded"
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    model_elapsed = time.time() - model_start
                    if os.getenv("ADV_FINBERT_VERBOSE", "0") == "1":
                        logger.info(f"[TIMING][advanced_sentiment][finbert_infer] {model_elapsed:.2f} seconds")
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    # Softmax to get probabilities
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    
                    # Check model label mapping (different models have different orders)
                    if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                        label_mapping = self.model.config.id2label
                        if label_mapping.get(0) == 'NEUTRAL' and label_mapping.get(1) == 'BULLISH':
                            # FinTwitBERT order: [NEUTRAL, BULLISH, BEARISH]
                            neu, bull, bear = float(probs[0]), float(probs[1]), float(probs[2])
                            if bull >= max(neu, bear):
                                label = "POSITIVE"
                                score_val = bull
                            elif bear >= max(bull, neu):
                                label = "NEGATIVE"
                                score_val = bear
                            else:
                                label = "NEUTRAL"
                                score_val = neu
                        elif label_mapping.get(0) == 'LABEL_0' and label_mapping.get(1) == 'LABEL_1':
                            # Twitter RoBERTa order: [NEGATIVE, NEUTRAL, POSITIVE] (LABEL_0=neg, LABEL_1=neu, LABEL_2=pos)
                            neg, neu, pos = float(probs[0]), float(probs[1]), float(probs[2])
                            if pos >= max(neg, neu):
                                label = "POSITIVE"
                                score_val = pos
                            elif neg >= max(pos, neu):
                                label = "NEGATIVE"
                                score_val = neg
                            else:
                                label = "NEUTRAL"
                                score_val = neu
                        else:
                            # Standard FinBERT order: [NEGATIVE, NEUTRAL, POSITIVE]
                            neg, neu, pos = float(probs[0]), float(probs[1]), float(probs[2])
                            if pos >= max(neg, neu):
                                label = "POSITIVE"
                                score_val = pos
                            elif neg >= max(pos, neu):
                                label = "NEGATIVE"
                                score_val = neg
                            else:
                                label = "NEUTRAL"
                                score_val = neu
                    else:
                        # Fallback to standard order
                        neg, neu, pos = float(probs[0]), float(probs[1]), float(probs[2])
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

                # Apply financial context boosting for earnings, guidance, etc.
                sentiment_score = self._apply_financial_context_boost(text, sentiment_score, confidence)

                return float(sentiment_score), confidence

            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = '429' in error_msg or 'too many requests' in error_msg or 'rate limit' in error_msg
                
                logger.warning(f"FinBERT analysis error (attempt {retries+1}): {e}")
                self.is_loaded = False  # Force reload next attempt
                if retries >= max_retries:
                    logger.error(f"FinBERT giving up after {retries+1} attempts: {e}")
                    return 0.0, 0.0
                
                # Use longer backoff for rate limiting
                if is_rate_limit:
                    backoff = min(2.0 * (2 ** retries), 30.0)  # Longer backoff for 429
                    logger.info(f"Rate limit detected, using extended backoff: {backoff}s")
                else:
                    backoff = min(0.5 * (2 ** retries), 2.0)  # Standard backoff for other errors
                
                time.sleep(backoff)
                self._load_model()
                retries += 1
    
    def _preprocess_financial_text(self, text: str) -> str:
        """Enhanced preprocessing for financial text analysis"""
        # Remove common financial noise patterns
        text = text.replace('$', '').replace('%', ' percent ')
        
        # Expand common financial abbreviations
        abbreviation_map = {
            'EPS': 'earnings per share',
            'EBITDA': 'earnings before interest taxes depreciation amortization',
            'P/E': 'price to earnings ratio',
            'ROI': 'return on investment',
            'IPO': 'initial public offering',
            'M&A': 'mergers and acquisitions',
            'CEO': 'chief executive officer',
            'CFO': 'chief financial officer',
            'Q1': 'first quarter',
            'Q2': 'second quarter',
            'Q3': 'third quarter',
            'Q4': 'fourth quarter'
        }
        
        for abbr, full in abbreviation_map.items():
            text = text.replace(abbr, full)
        
        return text
    
    def _apply_financial_context_boost(self, text: str, sentiment_score: float, confidence: float) -> float:
        """Apply context-aware boosting for financial-specific content"""
        text_lower = text.lower()
        boost_factor = 1.0
        
        # Earnings-related content gets higher confidence
        earnings_terms = {'earnings', 'results', 'quarter', 'guidance', 'forecast', 'outlook'}
        if any(term in text_lower for term in earnings_terms):
            boost_factor *= 1.15
            confidence = min(1.0, confidence * 1.1)
        
        # M&A and corporate action content
        corporate_terms = {'acquisition', 'merger', 'buyout', 'takeover', 'deal'}
        if any(term in text_lower for term in corporate_terms):
            boost_factor *= 1.1
        
        # Apply boost with confidence scaling
        boosted_score = sentiment_score * boost_factor
        return max(-1.0, min(1.0, boosted_score))  # Clamp to valid range

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
            finbert_sentiment, finbert_confidence = self.finbert_analyzer.analyze(text, source)
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
            dispersion = float(np.std(model_vals)) if len(model_vals) >= 2 else 0.0
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
            source=source,
        )

        processing_time = time.time() - start_time
        result.processing_time = processing_time
        # Throttle verbose per-text timing logs unless explicitly enabled
        if os.getenv("ADV_SENTIMENT_VERBOSE", "0") == "1":
            logger.info(f"[TIMING][advanced_sentiment][analyze_text] {processing_time:.2f}s for {symbol} source={source}")
            print(f"[TIMING][advanced_sentiment][analyze_text] {processing_time:.2f}s for {symbol} source={source}")

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

        results: List[SentimentResult] = []

        # Micro-batching FinBERT path: we first run lightweight VADER + lexicon sequentially,
        # collect batch of texts needing FinBERT, run FinBERT in a vectorized forward pass,
        # then assemble ensemble scores. This avoids per-text model invocations.
        batch_start = time.time()
        finbert_needed: List[Tuple[int, str]] = []  # (index, text)
        intermediate: List[Dict[str, Any]] = []

        for idx, (text, symbol, source) in enumerate(zip(texts, symbols, sources)):
            start_time = time.time()
            vader = self.vader_analyzer.polarity_scores(text)
            vader_compound = vader.get('compound', 0.0)
            lex_score, lex_matches = self.financial_lexicon.get_lexicon_score(text)
            # Heuristic: always include for FinBERT unless text extremely short
            if len(text.strip()) >= 8:
                finbert_needed.append((idx, text))
            intermediate.append({
                'symbol': symbol,
                'source': source or 'unknown',
                'text': text,
                'vader': vader_compound,
                'lex_score': lex_score,
                'lex_matches': lex_matches,
                'start_time': start_time
            })

        # Run FinBERT in batches (configurable size)
        finbert_scores: Dict[int, Tuple[float, float]] = {}
        batch_size = max(1, int(os.getenv('FINBERT_BATCH_SIZE', '8')))
        if finbert_needed:
            try:
                # Ensure model/tokenizer loaded
                if not self.finbert_analyzer.is_loaded:
                    self.finbert_analyzer._load_model()
                if self.finbert_analyzer.tokenizer is None or self.finbert_analyzer.model is None:
                    raise RuntimeError("FinBERT model/tokenizer not available for batch inference")
                for i in range(0, len(finbert_needed), batch_size):
                    chunk = finbert_needed[i:i+batch_size]
                    texts_chunk = [t for _, t in chunk]
                    # Determine dominant source for model selection
                    chunk_indices = [idx for idx, _ in chunk]
                    sources_in_chunk = [intermediate[idx]['source'] for idx in chunk_indices if idx < len(intermediate)]
                    dominant_source = max(set(sources_in_chunk), key=sources_in_chunk.count) if sources_in_chunk else 'unknown'
                    
                    # Ensure correct model for dominant source
                    self.finbert_analyzer._ensure_correct_model(dominant_source)
                    
                    # Tokenize as batch
                    with self.finbert_analyzer._inference_lock:
                        enc = self.finbert_analyzer.tokenizer(
                            texts_chunk,
                            add_special_tokens=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt',
                            padding=True
                        )
                    with self.finbert_analyzer._instance_lock, torch.no_grad():
                        outputs = self.finbert_analyzer.model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])  # type: ignore
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    for (orig_idx, _), row in zip(chunk, probs):
                        neg, neu, pos = float(row[0]), float(row[1]), float(row[2])
                        if pos >= max(neg, neu):
                            sentiment_score = pos
                            conf = pos
                        elif neg >= max(pos, neu):
                            sentiment_score = -neg
                            conf = neg
                        else:
                            sentiment_score = 0.0
                            conf = neu
                        finbert_scores[orig_idx] = (sentiment_score, conf)
            except Exception as e:
                logger.warning(f"FinBERT batch inference failed, falling back to per-text: {e}")
                for orig_idx, text in finbert_needed:
                    # Get source from intermediate data
                    source = intermediate[orig_idx]['source'] if orig_idx < len(intermediate) else 'unknown'
                    sc, cf = self.finbert_analyzer.analyze(text, source)
                    finbert_scores[orig_idx] = (sc, cf)

        # Assemble results
        for idx, meta in enumerate(intermediate):
            text = meta['text']
            symbol = meta['symbol']
            source = meta['source']
            vader_compound = meta['vader']
            lex_score = meta['lex_score']
            lex_matches = meta['lex_matches']
            start_time = meta['start_time']
            finbert_score, finbert_conf = finbert_scores.get(idx, (0.0, 0.0))

            # Derive dynamic weights using existing helper
            weights = self._get_dynamic_weights(len(text), lex_matches, finbert_conf)
            vader_w = float(weights.get('vader', 0.3))
            finbert_w = float(weights.get('finbert', 0.5))
            lex_w = float(weights.get('lexicon', 0.2))
            denom = vader_w + finbert_w + lex_w or 1.0
            numerator = vader_compound * vader_w + finbert_score * finbert_w + (lex_score if lex_matches > 0 else 0.0) * lex_w
            ensemble_score = numerator / denom

            # Confidence: blend FinBERT + VADER dispersion heuristic
            confidence = max(0.05, min(0.99, (abs(finbert_score) * 0.5 + abs(vader_compound) * 0.3 + min(1.0, lex_matches / 8.0) * 0.2)))
            processing_time = time.time() - start_time
            result = SentimentResult(
                symbol=symbol,
                text_snippet=text[:100] + "..." if len(text) > 100 else text,
                vader_score=vader_compound,
                finbert_score=finbert_score,
                ensemble_score=ensemble_score,
                confidence=confidence,
                timestamp=datetime.now(),
                model_weights=weights,
                text_length=len(text),
                source=source,
            )
            result.processing_time = processing_time
            results.append(result)
            if os.getenv('ADV_SENTIMENT_VERBOSE', '0') == '1':
                logger.info(f"[TIMING][advanced_sentiment][analyze_text_mb] {processing_time:.2f}s for {symbol} source={source}")

        batch_elapsed = time.time() - batch_start
        logger.info(f"[TIMING][advanced_sentiment][analyze_batch] {batch_elapsed:.2f}s for {len(texts)} texts")
        if os.getenv("ADV_SENTIMENT_VERBOSE", "0") == "1":
            print(f"[TIMING][advanced_sentiment][analyze_batch] {batch_elapsed:.2f}s for {len(texts)} texts")

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
        quality_score = self._calculate_enhanced_quality_score(results, avg_confidence)

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
    
    def _calculate_quality_score(self, sentiment_results: Dict[str, SentimentResult], average_confidence: float) -> float:
        """Calculate quality score based on source diversity and confidence"""
        source_diversity_bonus = min(20, len(sentiment_results) * 3)  # Max 20 points for source diversity
        confidence_score = average_confidence * 60  # Max 60 points for confidence
        sample_size_bonus = min(20, sum(sd.sample_size or 0 for sd in sentiment_results.values()) / 10)  # Max 20 points for sample size

        return source_diversity_bonus + confidence_score + sample_size_bonus

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
    
    def _calculate_enhanced_quality_score(self, results: List[SentimentResult], avg_confidence: float) -> float:
        """Calculate enhanced quality score with multiple factors"""
        if not results:
            return 0.0
        
        # Base confidence score (40% weight)
        confidence_score = min(40.0, avg_confidence * 40.0)
        
        # Source diversity bonus (20% weight)
        unique_sources = len(set(r.source for r in results))
        diversity_score = min(20.0, unique_sources * 4.0)
        
        # Sample size bonus (15% weight)
        total_samples = len(results)
        sample_score = min(15.0, total_samples * 1.5)
        
        # Model agreement bonus (15% weight)
        model_scores = [r.ensemble_score for r in results if abs(r.ensemble_score) > 0.01]
        if len(model_scores) >= 2:
            # Calculate dispersion (lower dispersion = higher agreement)
            dispersion = float(np.std(model_scores)) if len(model_scores) > 1 else 0.0
            agreement_score = max(0.0, 15.0 * (1.0 - min(1.0, dispersion * 2.0)))
        else:
            agreement_score = 7.5  # Neutral score for single result
        
        # Content quality bonus (10% weight)
        avg_text_length = sum(len(r.text_snippet) for r in results) / len(results)
        content_score = min(10.0, max(0.0, (avg_text_length - 50) / 5.0))  # Bonus for longer, relevant content
        
        total_score = confidence_score + diversity_score + sample_score + agreement_score + content_score
        
        return min(100.0, max(0.0, total_score))

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

def fetch_sentiment_data(tickers: List[str]) -> List[SentimentSummary]:
    """
    Legacy compatibility function for prompt_chain.py and data feeds.
    Returns sentiment summaries for given tickers using empty text lists.
    """
    engine = get_sentiment_engine()
    return [engine.get_symbol_sentiment_summary(ticker, []) for ticker in tickers]