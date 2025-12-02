"""
ORACLE-X Advanced Prompt Optimization Engine

This module implements a self-improving prompt generation and optimization system
that adapts prompts based on market conditions, performance feedback, and
intelligent context prioritization.

Key Features:
- Dynamic prompt templates based on market conditions
- Intelligent signal prioritization using semantic analysis
- Performance-based prompt evolution
- A/B testing framework for systematic optimization
- Real-time context management and token budget optimization
"""

import hashlib
import json
import logging
import random
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Market condition classifications for prompt adaptation"""

    BULLISH = "bullish"
    BEARISH = "bearish"
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"
    EARNINGS = "earnings"
    NEWS_DRIVEN = "news_driven"
    OPTIONS_HEAVY = "options_heavy"
    LOW_VOLUME = "low_volume"


class PromptStrategy(Enum):
    """Different prompt generation strategies"""

    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


@dataclass
class PromptPerformance:
    """Track performance metrics for prompt variants"""

    prompt_id: str
    template_id: str
    market_condition: MarketCondition
    strategy: PromptStrategy
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_latency: float = 0.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

    # Trading performance (if available)
    avg_profit_target_hit: float = 0.0
    avg_stop_loss_hit: float = 0.0
    avg_trade_duration: float = 0.0


@dataclass
class PromptTemplate:
    """Advanced prompt template with dynamic capabilities"""

    template_id: str
    name: str
    strategy: PromptStrategy
    market_conditions: List[MarketCondition]
    system_prompt: str
    user_prompt_template: str
    max_tokens: int = 1024
    temperature: float = 0.3
    priority_signals: List[str] = field(default_factory=list)
    signal_weights: Dict[str, float] = field(default_factory=dict)
    context_compression_ratio: float = 0.8
    performance: Optional[PromptPerformance] = None


class PromptOptimizationEngine:
    """
    Core engine for prompt optimization with self-learning capabilities
    """

    def __init__(self, db_path: str = "data/databases/prompt_optimization.db"):
        self.db_path = db_path
        self.performance_history = deque(maxlen=1000)  # Keep recent performance data
        self.prompt_templates = {}
        self.active_experiments = {}
        self.mutation_strategies = [
            self._mutate_signal_weights,
            self._mutate_temperature,
            self._mutate_max_tokens,
            self._mutate_prompt_structure,
            self._mutate_context_compression,
        ]
        self._init_database()
        self._load_default_templates()

    def _init_database(self):
        """Initialize SQLite database for performance tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prompt_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prompt_id TEXT NOT NULL,
                        template_id TEXT NOT NULL,
                        market_condition TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        success_rate REAL DEFAULT 0.0,
                        avg_confidence REAL DEFAULT 0.0,
                        avg_latency REAL DEFAULT 0.0,
                        usage_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_used TIMESTAMP,
                        performance_data TEXT,  -- JSON blob for additional metrics
                        UNIQUE(prompt_id)
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prompt_experiments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id TEXT NOT NULL,
                        variant_a_id TEXT NOT NULL,
                        variant_b_id TEXT NOT NULL,
                        market_condition TEXT NOT NULL,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        variant_a_performance REAL DEFAULT 0.0,
                        variant_b_performance REAL DEFAULT 0.0,
                        statistical_significance REAL DEFAULT 0.0,
                        winner TEXT,
                        status TEXT DEFAULT 'active'
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS signal_importance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_name TEXT NOT NULL,
                        market_condition TEXT NOT NULL,
                        importance_score REAL DEFAULT 0.0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(signal_name, market_condition)
                    )
                """
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _load_default_templates(self):
        """Load default prompt templates for different strategies"""
        templates = [
            PromptTemplate(
                template_id="conservative_balanced",
                name="Conservative Balanced Strategy",
                strategy=PromptStrategy.CONSERVATIVE,
                market_conditions=[
                    MarketCondition.SIDEWAYS,
                    MarketCondition.LOW_VOLUME,
                ],
                system_prompt="""You are ORACLE-X, a conservative trading scenario engine focused on high-probability, low-risk opportunities. Prioritize capital preservation and consistent returns over aggressive gains.""",
                user_prompt_template="""
Analyze the following market data with a conservative approach:

{signals_section}

Generate 1-2 high-confidence trades with:
- Minimum 75% confidence level
- Clear risk management (stop-loss within 3-5%)
- Conservative profit targets (3-8% gains)
- Strong technical confirmation signals
- Detailed risk assessment for each trade

Focus on established patterns and avoid speculative positions.
""",
                priority_signals=[
                    "market_internals",
                    "chart_analysis",
                    "finviz_breadth",
                ],
                signal_weights={
                    "market_internals": 0.3,
                    "chart_analysis": 0.25,
                    "options_flow": 0.15,
                    "sentiment_llm": 0.15,
                    "finviz_breadth": 0.15,
                },
            ),
            PromptTemplate(
                template_id="aggressive_momentum",
                name="Aggressive Momentum Strategy",
                strategy=PromptStrategy.AGGRESSIVE,
                market_conditions=[MarketCondition.BULLISH, MarketCondition.VOLATILE],
                system_prompt="""You are ORACLE-X, an aggressive momentum trading engine designed to capitalize on strong directional moves and high-volatility opportunities.""",
                user_prompt_template="""
Analyze market momentum and volatility signals:

{signals_section}

Generate 2-3 aggressive momentum trades:
- Focus on strong directional moves (>5% expected)
- Utilize options strategies for leverage when appropriate
- Higher risk tolerance (stop-loss 5-10%)
- Capitalize on volatility spikes and momentum breaks
- Include both short-term (1-3 days) and swing positions

Prioritize momentum confirmation and volume validation.
""",
                priority_signals=[
                    "options_flow",
                    "dark_pools",
                    "sentiment_web",
                    "chart_analysis",
                ],
                signal_weights={
                    "options_flow": 0.3,
                    "dark_pools": 0.2,
                    "sentiment_web": 0.2,
                    "chart_analysis": 0.2,
                    "market_internals": 0.1,
                },
            ),
            PromptTemplate(
                template_id="earnings_specialist",
                name="Earnings Event Strategy",
                strategy=PromptStrategy.FUNDAMENTAL,
                market_conditions=[
                    MarketCondition.EARNINGS,
                    MarketCondition.NEWS_DRIVEN,
                ],
                system_prompt="""You are ORACLE-X, specialized in earnings and news-driven trading opportunities. Focus on event-driven catalysts and fundamental analysis.""",
                user_prompt_template="""
Analyze earnings and news catalysts:

{signals_section}

Generate earnings-focused trades:
- Pre/post earnings strategies
- Options strategies for volatility plays
- News catalyst identification
- Fundamental analysis integration
- Event timing considerations

Include both long and short opportunities based on expectations vs reality.
""",
                priority_signals=[
                    "earnings_calendar",
                    "sentiment_web",
                    "yahoo_headlines",
                    "options_flow",
                ],
                signal_weights={
                    "earnings_calendar": 0.35,
                    "sentiment_web": 0.25,
                    "yahoo_headlines": 0.2,
                    "options_flow": 0.2,
                },
            ),
            PromptTemplate(
                template_id="technical_precision",
                name="Technical Analysis Focus",
                strategy=PromptStrategy.TECHNICAL,
                market_conditions=[MarketCondition.SIDEWAYS, MarketCondition.VOLATILE],
                system_prompt="""You are ORACLE-X, a technical analysis specialist. Focus on chart patterns, support/resistance levels, and technical indicators.""",
                user_prompt_template="""
Technical analysis focus:

{signals_section}

Generate technically-driven trades:
- Chart pattern recognition
- Support/resistance level analysis
- Technical indicator confirmation
- Risk/reward ratio optimization
- Entry/exit precision timing

Emphasize technical confluence and pattern reliability.
""",
                priority_signals=[
                    "chart_analysis",
                    "market_internals",
                    "finviz_breadth",
                ],
                signal_weights={
                    "chart_analysis": 0.4,
                    "market_internals": 0.3,
                    "finviz_breadth": 0.2,
                    "options_flow": 0.1,
                },
                temperature=0.2,  # Lower temperature for more consistent technical analysis
            ),
        ]

        for template in templates:
            self.prompt_templates[template.template_id] = template

    def classify_market_condition(self, signals: Dict[str, Any]) -> MarketCondition:
        """
        Intelligently classify current market conditions based on signals
        """
        try:
            # Extract key indicators
            market_internals = str(signals.get("market_internals", "")).lower()
            options_flow = signals.get("options_flow", [])
            sentiment_llm = str(signals.get("sentiment_llm", "")).lower()
            finviz_breadth = str(signals.get("finviz_breadth", "")).lower()

            # Check for earnings events
            earnings_calendar = signals.get("earnings_calendar", [])
            if earnings_calendar and len(earnings_calendar) > 3:
                return MarketCondition.EARNINGS

            # Check for high options activity
            if len(options_flow) > 20:
                return MarketCondition.OPTIONS_HEAVY

            # Sentiment analysis
            bullish_keywords = [
                "bullish",
                "optimistic",
                "positive",
                "rally",
                "breakout",
            ]
            bearish_keywords = [
                "bearish",
                "pessimistic",
                "negative",
                "decline",
                "breakdown",
            ]

            bullish_score = sum(1 for word in bullish_keywords if word in sentiment_llm)
            bearish_score = sum(1 for word in bearish_keywords if word in sentiment_llm)

            # Volatility indicators
            volatile_keywords = [
                "volatile",
                "uncertainty",
                "swing",
                "whipsaw",
                "erratic",
            ]
            volatility_score = sum(
                1
                for word in volatile_keywords
                if word in market_internals + sentiment_llm
            )

            if volatility_score > 2:
                return MarketCondition.VOLATILE
            elif bullish_score > bearish_score and bullish_score > 1:
                return MarketCondition.BULLISH
            elif bearish_score > bullish_score and bearish_score > 1:
                return MarketCondition.BEARISH
            else:
                return MarketCondition.SIDEWAYS

        except Exception as e:
            logger.warning(f"Failed to classify market condition: {e}")
            return MarketCondition.SIDEWAYS

    def select_optimal_template(
        self,
        market_condition: MarketCondition,
        recent_performance: Optional[Dict] = None,
    ) -> PromptTemplate:
        """
        Select the best performing template for current market conditions
        """
        # Filter templates by market condition compatibility
        compatible_templates = [
            template
            for template in self.prompt_templates.values()
            if market_condition in template.market_conditions
            or not template.market_conditions
        ]

        if not compatible_templates:
            # Fallback to default conservative template
            return self.prompt_templates.get(
                "conservative_balanced", list(self.prompt_templates.values())[0]
            )

        # If we have performance data, select based on historical success
        if recent_performance:
            best_template = max(
                compatible_templates,
                key=lambda t: recent_performance.get(t.template_id, {}).get(
                    "success_rate", 0.5
                ),
            )
            return best_template

        # Fallback: select based on strategy appropriateness
        strategy_priority = {
            MarketCondition.BULLISH: [
                PromptStrategy.AGGRESSIVE,
                PromptStrategy.MOMENTUM,
            ],
            MarketCondition.BEARISH: [
                PromptStrategy.CONTRARIAN,
                PromptStrategy.CONSERVATIVE,
            ],
            MarketCondition.VOLATILE: [
                PromptStrategy.AGGRESSIVE,
                PromptStrategy.TECHNICAL,
            ],
            MarketCondition.SIDEWAYS: [
                PromptStrategy.CONSERVATIVE,
                PromptStrategy.TECHNICAL,
            ],
            MarketCondition.EARNINGS: [
                PromptStrategy.FUNDAMENTAL,
                PromptStrategy.AGGRESSIVE,
            ],
            MarketCondition.NEWS_DRIVEN: [
                PromptStrategy.FUNDAMENTAL,
                PromptStrategy.MOMENTUM,
            ],
            MarketCondition.OPTIONS_HEAVY: [
                PromptStrategy.AGGRESSIVE,
                PromptStrategy.TECHNICAL,
            ],
            MarketCondition.LOW_VOLUME: [
                PromptStrategy.CONSERVATIVE,
                PromptStrategy.TECHNICAL,
            ],
        }

        preferred_strategies = strategy_priority.get(
            market_condition, [PromptStrategy.BALANCED]
        )

        for strategy in preferred_strategies:
            for template in compatible_templates:
                if template.strategy == strategy:
                    return template

        return compatible_templates[0]  # Fallback to first compatible

    def optimize_signal_selection(
        self,
        signals: Dict[str, Any],
        template: PromptTemplate,
        token_budget: int = 3000,
    ) -> Dict[str, Any]:
        """
        Intelligently select and prioritize signals based on template preferences and token budget
        """
        optimized_signals = {}
        current_tokens = 0

        # Sort signals by priority weights from template
        signal_priorities = []
        for signal_name, signal_data in signals.items():
            weight = template.signal_weights.get(signal_name, 0.1)
            priority = weight * (
                1.0 if signal_name in template.priority_signals else 0.5
            )
            signal_priorities.append((signal_name, signal_data, priority))

        # Sort by priority (descending)
        signal_priorities.sort(key=lambda x: x[2], reverse=True)

        # Add signals within token budget
        for signal_name, signal_data, priority in signal_priorities:
            signal_tokens = self._estimate_tokens(signal_data)

            if current_tokens + signal_tokens <= token_budget:
                optimized_signals[signal_name] = signal_data
                current_tokens += signal_tokens
            else:
                # Try to compress the signal if it's high priority
                if priority > 0.5:
                    compressed = self._compress_signal(
                        signal_data, max_tokens=token_budget - current_tokens
                    )
                    if compressed:
                        optimized_signals[signal_name] = compressed
                        current_tokens += self._estimate_tokens(compressed)

        return optimized_signals

    def _estimate_tokens(self, data: Any) -> int:
        """Rough token estimation for data"""
        if isinstance(data, str):
            return int(len(data.split()) * 1.3)  # Rough approximation
        elif isinstance(data, (list, dict)):
            return len(str(data)) // 4  # Very rough approximation
        else:
            return len(str(data)) // 4

    def _compress_signal(self, signal_data: Any, max_tokens: int = 100) -> Any:
        """Compress signal data to fit within token budget"""
        if isinstance(signal_data, str):
            words = signal_data.split()
            if len(words) > max_tokens:
                return " ".join(words[:max_tokens]) + "..."
            return signal_data
        elif isinstance(signal_data, list):
            if len(signal_data) > 5:
                return signal_data[:5]  # Take top 5 items
            return signal_data
        elif isinstance(signal_data, dict):
            # Keep only top-level keys and compress values
            compressed = {}
            for k, v in list(signal_data.items())[:3]:  # Top 3 keys
                if isinstance(v, str) and len(v) > 50:
                    compressed[k] = v[:50] + "..."
                else:
                    compressed[k] = v
            return compressed
        return signal_data

    def generate_optimized_prompt(
        self,
        signals: Dict[str, Any],
        market_condition: Optional[MarketCondition] = None,
        template_id: Optional[str] = None,
    ) -> Tuple[str, str, Dict]:
        """
        Generate an optimized prompt based on current conditions and performance history
        """
        # Classify market condition if not provided
        if market_condition is None:
            market_condition = self.classify_market_condition(signals)

        # Select optimal template
        if template_id:
            template = self.prompt_templates.get(template_id)
            if not template:
                logger.warning(f"Template {template_id} not found, selecting optimal")
                template = self.select_optimal_template(market_condition)
        else:
            template = self.select_optimal_template(market_condition)

        # Optimize signal selection
        optimized_signals = self.optimize_signal_selection(signals, template)

        # Format signals section
        signals_section = self._format_signals_section(optimized_signals, template)

        # Generate final prompt
        user_prompt = template.user_prompt_template.format(
            signals_section=signals_section
        )

        # Create metadata for tracking
        prompt_metadata = {
            "template_id": template.template_id,
            "market_condition": market_condition.value,
            "strategy": template.strategy.value,
            "signal_count": len(optimized_signals),
            "estimated_tokens": self._estimate_tokens(user_prompt),
            "timestamp": datetime.now().isoformat(),
        }

        return template.system_prompt, user_prompt, prompt_metadata

    def _format_signals_section(
        self, signals: Dict[str, Any], template: PromptTemplate
    ) -> str:
        """Format signals section with intelligent prioritization"""
        sections = []

        for signal_name, signal_data in signals.items():
            weight = template.signal_weights.get(signal_name, 0.1)

            # Format based on signal type
            if signal_name == "market_internals":
                sections.append(
                    f"**Market Internals** (Weight: {weight:.1f}):\n{signal_data}\n"
                )
            elif signal_name == "options_flow":
                if isinstance(signal_data, list) and signal_data:
                    formatted = "\n".join([f"  - {item}" for item in signal_data[:5]])
                    sections.append(
                        f"**Options Flow** (Weight: {weight:.1f}):\n{formatted}\n"
                    )
            elif signal_name == "sentiment_llm":
                sections.append(
                    f"**LLM Sentiment** (Weight: {weight:.1f}):\n{signal_data}\n"
                )
            elif signal_name == "chart_analysis":
                sections.append(
                    f"**Chart Analysis** (Weight: {weight:.1f}):\n{signal_data}\n"
                )
            else:
                # Generic formatting
                if isinstance(signal_data, (list, dict)):
                    formatted = str(signal_data)[:200] + (
                        "..." if len(str(signal_data)) > 200 else ""
                    )
                else:
                    formatted = str(signal_data)[:200] + (
                        "..." if len(str(signal_data)) > 200 else ""
                    )
                sections.append(
                    f"**{signal_name.replace('_', ' ').title()}** (Weight: {weight:.1f}):\n{formatted}\n"
                )

        return "\n".join(sections)

    def record_prompt_performance(
        self,
        prompt_metadata: Dict,
        model_attempts: List[Dict],
        trading_outcome: Optional[Dict] = None,
    ):
        """Record performance data for prompt optimization"""
        try:
            prompt_id = self._generate_prompt_id(prompt_metadata)

            # Calculate performance metrics
            success_rate = (
                sum(1 for attempt in model_attempts if attempt.get("success", False))
                / len(model_attempts)
                if model_attempts
                else 0.0
            )
            avg_latency = (
                sum(attempt.get("latency_sec", 0) for attempt in model_attempts)
                / len(model_attempts)
                if model_attempts
                else 0.0
            )

            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO prompt_performance 
                    (prompt_id, template_id, market_condition, strategy, success_rate, avg_latency, usage_count, last_used, performance_data)
                    VALUES (?, ?, ?, ?, ?, ?, 
                            COALESCE((SELECT usage_count FROM prompt_performance WHERE prompt_id = ?), 0) + 1,
                            ?, ?)
                """,
                    (
                        prompt_id,
                        prompt_metadata["template_id"],
                        prompt_metadata["market_condition"],
                        prompt_metadata["strategy"],
                        success_rate,
                        avg_latency,
                        prompt_id,  # for the COALESCE query
                        datetime.now(),
                        json.dumps(
                            {
                                "model_attempts": model_attempts,
                                "trading_outcome": trading_outcome,
                                "metadata": prompt_metadata,
                            }
                        ),
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to record prompt performance: {e}")

    def _generate_prompt_id(self, metadata: Dict) -> str:
        """Generate unique ID for prompt variant"""
        key_data = f"{metadata['template_id']}_{metadata['market_condition']}_{metadata['strategy']}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]

    def start_ab_test(
        self,
        template_a_id: str,
        template_b_id: str,
        market_condition: MarketCondition,
        duration_hours: int = 24,
    ) -> str:
        """Start A/B test between two prompt templates"""
        experiment_id = f"exp_{int(time.time())}_{random.randint(1000, 9999)}"

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO prompt_experiments 
                    (experiment_id, variant_a_id, variant_b_id, market_condition, end_time)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        experiment_id,
                        template_a_id,
                        template_b_id,
                        market_condition.value,
                        datetime.now() + timedelta(hours=duration_hours),
                    ),
                )
                conn.commit()

            self.active_experiments[experiment_id] = {
                "variant_a": template_a_id,
                "variant_b": template_b_id,
                "market_condition": market_condition,
                "start_time": datetime.now(),
                "results_a": [],
                "results_b": [],
            }

            logger.info(
                f"Started A/B test {experiment_id}: {template_a_id} vs {template_b_id}"
            )
            return experiment_id

        except Exception as e:
            logger.error(f"Failed to start A/B test: {e}")
            return ""

    def evolve_prompts(
        self, top_performers: List[str], mutation_rate: float = 0.1
    ) -> List[PromptTemplate]:
        """Use genetic algorithm principles to evolve prompt templates"""
        new_templates = []

        for template_id in top_performers:
            parent_template = self.prompt_templates.get(template_id)
            if not parent_template:
                continue

            # Create mutations
            for i in range(2):  # Create 2 mutations per parent
                mutated = self._mutate_template(parent_template, mutation_rate)
                if mutated:
                    new_templates.append(mutated)

        return new_templates

    def _mutate_template(
        self, template: PromptTemplate, mutation_rate: float
    ) -> Optional[PromptTemplate]:
        """Create a mutated version of a prompt template"""
        try:
            # Create a copy
            new_template = PromptTemplate(
                template_id=f"{template.template_id}_mut_{int(time.time())}",
                name=f"{template.name} (Evolved)",
                strategy=template.strategy,
                market_conditions=template.market_conditions.copy(),
                system_prompt=template.system_prompt,
                user_prompt_template=template.user_prompt_template,
                max_tokens=template.max_tokens,
                temperature=template.temperature,
                priority_signals=template.priority_signals.copy(),
                signal_weights=template.signal_weights.copy(),
                context_compression_ratio=template.context_compression_ratio,
            )

            # Apply random mutations
            num_mutations = max(1, int(len(self.mutation_strategies) * mutation_rate))
            selected_mutations = random.sample(self.mutation_strategies, num_mutations)

            for mutation_func in selected_mutations:
                mutation_func(new_template)

            return new_template

        except Exception as e:
            logger.error(f"Failed to mutate template: {e}")
            return None

    def _mutate_signal_weights(self, template: PromptTemplate):
        """Mutate signal weights"""
        for signal in template.signal_weights:
            if random.random() < 0.3:  # 30% chance to mutate each weight
                current_weight = template.signal_weights[signal]
                mutation = random.uniform(-0.1, 0.1)
                template.signal_weights[signal] = max(
                    0.0, min(1.0, current_weight + mutation)
                )

    def _mutate_temperature(self, template: PromptTemplate):
        """Mutate temperature parameter"""
        mutation = random.uniform(-0.1, 0.1)
        template.temperature = max(0.1, min(0.9, template.temperature + mutation))

    def _mutate_max_tokens(self, template: PromptTemplate):
        """Mutate max tokens"""
        mutation = random.randint(-200, 200)
        template.max_tokens = max(512, min(2048, template.max_tokens + mutation))

    def _mutate_prompt_structure(self, template: PromptTemplate):
        """Mutate prompt structure elements"""
        # Add randomness to prompt phrasing
        variations = [
            ("Generate", "Create"),
            ("Analyze", "Examine"),
            ("Focus on", "Prioritize"),
            ("Include", "Incorporate"),
            ("trades", "opportunities"),
            ("confidence", "probability"),
        ]

        for old, new in variations:
            if random.random() < 0.2:  # 20% chance for each variation
                template.user_prompt_template = template.user_prompt_template.replace(
                    old, new
                )

    def _mutate_context_compression(self, template: PromptTemplate):
        """Mutate context compression ratio"""
        mutation = random.uniform(-0.1, 0.1)
        template.context_compression_ratio = max(
            0.3, min(1.0, template.context_compression_ratio + mutation)
        )

    def get_performance_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Template performance
                cursor = conn.execute(
                    """
                    SELECT template_id, market_condition, strategy,
                           AVG(success_rate) as avg_success,
                           AVG(avg_latency) as avg_latency,
                           SUM(usage_count) as total_usage
                    FROM prompt_performance 
                    WHERE last_used >= datetime('now', '-{} days')
                    GROUP BY template_id, market_condition, strategy
                    ORDER BY avg_success DESC
                """.format(
                        days
                    )
                )

                template_stats = cursor.fetchall()

                # Experiment results
                cursor = conn.execute(
                    """
                    SELECT experiment_id, variant_a_id, variant_b_id, 
                           variant_a_performance, variant_b_performance, winner
                    FROM prompt_experiments 
                    WHERE start_time >= datetime('now', '-{} days')
                    AND status = 'completed'
                """.format(
                        days
                    )
                )

                experiment_stats = cursor.fetchall()

                return {
                    "template_performance": [
                        {
                            "template_id": row[0],
                            "market_condition": row[1],
                            "strategy": row[2],
                            "avg_success_rate": row[3],
                            "avg_latency": row[4],
                            "total_usage": row[5],
                        }
                        for row in template_stats
                    ],
                    "experiment_results": [
                        {
                            "experiment_id": row[0],
                            "variant_a": row[1],
                            "variant_b": row[2],
                            "performance_a": row[3],
                            "performance_b": row[4],
                            "winner": row[5],
                        }
                        for row in experiment_stats
                    ],
                    "total_templates": len(self.prompt_templates),
                    "active_experiments": len(self.active_experiments),
                }

        except Exception as e:
            logger.error(f"Failed to get performance analytics: {e}")
            return {}


# Global instance for easy access
_optimization_engine = None


def get_optimization_engine() -> PromptOptimizationEngine:
    """Get global optimization engine instance"""
    global _optimization_engine
    if _optimization_engine is None:
        _optimization_engine = PromptOptimizationEngine()
    return _optimization_engine
