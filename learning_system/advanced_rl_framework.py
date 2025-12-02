"""
Advanced Reinforcement Learning Framework for Oracle-X
Multi-Agent Deep RL with Continuous Learning and Risk Management
"""

import logging
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from pathlib import Path
import threading
import warnings

warnings.filterwarnings("ignore")

# Try to import advanced ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical, Normal

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Create dummy classes
    class nn:
        class Module:
            pass


logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for reinforcement learning system"""

    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    memory_size: int = 10000
    batch_size: int = 64
    update_frequency: int = 100
    target_update_frequency: int = 1000
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True
    use_prioritized_replay: bool = True


@dataclass
class TradingEnvironment:
    """Trading environment state representation"""

    price_history: np.ndarray
    technical_indicators: Dict[str, float]
    market_sentiment: float
    volatility: float
    volume: float
    position: float  # Current position size
    cash: float
    portfolio_value: float
    risk_metrics: Dict[str, float]
    timestamp: datetime


@dataclass
class TradingAction:
    """Trading action representation"""

    action_type: str  # 'buy', 'sell', 'hold'
    position_size: float  # Percentage of portfolio
    confidence: float
    reasoning: str


class DuelingDQN(nn.Module if TORCH_AVAILABLE else object):
    """Dueling Deep Q-Network for trading decisions"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        if not TORCH_AVAILABLE:
            return

        super(DuelingDQN, self).__init__()

        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

    def forward(self, state):
        if not TORCH_AVAILABLE:
            return np.zeros(3)

        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with prioritized sampling"""
        if len(self.buffer) < batch_size:
            return None, None, None

        priorities = self.priorities[: len(self.buffer)]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class MultiAgentRLSystem:
    """Multi-agent reinforcement learning system for trading"""

    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()

        # Agent networks
        self.agents = {}
        self.target_networks = {}
        self.optimizers = {}
        self.replay_buffers = {}

        # Training state
        self.training_step = 0
        self.episode_rewards = defaultdict(list)
        self.performance_history = defaultdict(list)

        # Multi-agent coordination
        self.agent_weights = {}
        self.coordination_matrix = {}

        logger.info("Multi-agent RL system initialized")

    def create_agent(self, agent_name: str, state_size: int, action_size: int):
        """Create a new RL agent"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using dummy agent")
            self.agents[agent_name] = DummyAgent()
            return

        # Main network
        self.agents[agent_name] = DuelingDQN(state_size, action_size)

        # Target network
        self.target_networks[agent_name] = DuelingDQN(state_size, action_size)
        self.target_networks[agent_name].load_state_dict(
            self.agents[agent_name].state_dict()
        )

        # Optimizer
        self.optimizers[agent_name] = optim.Adam(
            self.agents[agent_name].parameters(), lr=self.config.learning_rate
        )

        # Replay buffer
        self.replay_buffers[agent_name] = PrioritizedReplayBuffer(
            self.config.memory_size
        )

        # Initialize weights
        self.agent_weights[agent_name] = 1.0

        logger.info(f"Created RL agent: {agent_name}")

    def get_action(
        self, agent_name: str, state: np.ndarray, training: bool = True
    ) -> int:
        """Get action from agent using epsilon-greedy policy"""
        if agent_name not in self.agents:
            return 1  # Default hold action

        if not TORCH_AVAILABLE:
            return np.random.choice(3)  # Random action

        agent = self.agents[agent_name]

        if training and np.random.random() < self.config.epsilon:
            return np.random.choice(3)  # Random exploration

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent(state_tensor)
            return q_values.argmax().item()

    def store_experience(
        self, agent_name: str, state, action, reward, next_state, done
    ):
        """Store experience in agent's replay buffer"""
        if agent_name in self.replay_buffers:
            self.replay_buffers[agent_name].add(state, action, reward, next_state, done)

    def train_agent(self, agent_name: str):
        """Train individual agent"""
        if not TORCH_AVAILABLE or agent_name not in self.agents:
            return 0.0

        replay_buffer = self.replay_buffers[agent_name]
        experiences, indices, weights = replay_buffer.sample(self.config.batch_size)

        if experiences is None:
            return 0.0

        # Prepare batch
        states = torch.FloatTensor([e[0] for e in experiences])
        actions = torch.LongTensor([e[1] for e in experiences])
        rewards = torch.FloatTensor([e[2] for e in experiences])
        next_states = torch.FloatTensor([e[3] for e in experiences])
        dones = torch.BoolTensor([e[4] for e in experiences])
        weights_tensor = torch.FloatTensor(weights)

        # Current Q values
        current_q_values = self.agents[agent_name](states).gather(
            1, actions.unsqueeze(1)
        )

        # Next Q values (Double DQN)
        if self.config.use_double_dqn:
            next_actions = self.agents[agent_name](next_states).argmax(1)
            next_q_values = self.target_networks[agent_name](next_states).gather(
                1, next_actions.unsqueeze(1)
            )
        else:
            next_q_values = (
                self.target_networks[agent_name](next_states).max(1)[0].unsqueeze(1)
            )

        # Target Q values
        target_q_values = rewards.unsqueeze(1) + (
            self.config.gamma * next_q_values * ~dones.unsqueeze(1)
        )

        # Loss calculation with importance sampling
        td_errors = current_q_values - target_q_values.detach()
        loss = (weights_tensor.unsqueeze(1) * td_errors.pow(2)).mean()

        # Optimize
        self.optimizers[agent_name].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent_name].parameters(), 1.0)
        self.optimizers[agent_name].step()

        # Update priorities
        priorities = td_errors.abs().detach().numpy().flatten() + 1e-6
        replay_buffer.update_priorities(indices, priorities)

        return loss.item()

    def update_target_networks(self):
        """Update target networks"""
        if not TORCH_AVAILABLE:
            return

        for agent_name in self.agents:
            if agent_name in self.target_networks:
                self.target_networks[agent_name].load_state_dict(
                    self.agents[agent_name].state_dict()
                )

    def coordinate_agents(
        self, market_state: TradingEnvironment
    ) -> Dict[str, TradingAction]:
        """Coordinate multiple agents for ensemble decision making"""
        agent_actions = {}
        agent_confidences = {}

        # Get individual agent actions
        state_vector = self._environment_to_state_vector(market_state)

        for agent_name in self.agents:
            action_idx = self.get_action(agent_name, state_vector, training=False)
            confidence = self._calculate_action_confidence(agent_name, state_vector)

            action = self._action_index_to_trading_action(action_idx, confidence)
            agent_actions[agent_name] = action
            agent_confidences[agent_name] = confidence

        # Coordinate using weighted voting
        coordinated_actions = self._weighted_action_coordination(
            agent_actions, agent_confidences
        )

        return coordinated_actions

    def _environment_to_state_vector(self, env: TradingEnvironment) -> np.ndarray:
        """Convert environment to state vector"""
        state_features = []

        # Price features
        if len(env.price_history) > 0:
            state_features.extend(
                [
                    env.price_history[-1] if len(env.price_history) > 0 else 0,
                    (
                        np.mean(env.price_history[-5:])
                        if len(env.price_history) >= 5
                        else 0
                    ),
                    (
                        np.std(env.price_history[-10:])
                        if len(env.price_history) >= 10
                        else 0
                    ),
                ]
            )
        else:
            state_features.extend([0, 0, 0])

        # Technical indicators
        state_features.extend(
            [
                env.technical_indicators.get("rsi", 50) / 100,
                env.technical_indicators.get("macd", 0) / 10,
                env.technical_indicators.get("bb_position", 0.5),
            ]
        )

        # Market features
        state_features.extend(
            [
                env.market_sentiment,
                env.volatility,
                env.volume / 1000000,  # Normalize volume
                env.position,
                env.cash / env.portfolio_value if env.portfolio_value > 0 else 0,
            ]
        )

        # Risk metrics
        state_features.extend(
            [
                env.risk_metrics.get("sharpe_ratio", 0),
                env.risk_metrics.get("max_drawdown", 0),
                env.risk_metrics.get("var_95", 0),
            ]
        )

        return np.array(state_features, dtype=np.float32)

    def _action_index_to_trading_action(
        self, action_idx: int, confidence: float
    ) -> TradingAction:
        """Convert action index to trading action"""
        action_map = {0: ("sell", -0.1), 1: ("hold", 0.0), 2: ("buy", 0.1)}

        action_type, base_size = action_map.get(action_idx, ("hold", 0.0))

        # Adjust position size based on confidence
        position_size = base_size * confidence if action_type != "hold" else 0.0

        return TradingAction(
            action_type=action_type,
            position_size=position_size,
            confidence=confidence,
            reasoning=f"RL agent decision with {confidence:.2f} confidence",
        )

    def _calculate_action_confidence(self, agent_name: str, state: np.ndarray) -> float:
        """Calculate confidence in agent's action"""
        if not TORCH_AVAILABLE or agent_name not in self.agents:
            return 0.5

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.agents[agent_name](state_tensor)

            # Confidence based on Q-value spread
            q_max = q_values.max().item()
            q_min = q_values.min().item()
            q_spread = q_max - q_min

            # Normalize confidence
            confidence = min(1.0, q_spread / 10.0)
            return max(0.1, confidence)

    def _weighted_action_coordination(
        self,
        agent_actions: Dict[str, TradingAction],
        agent_confidences: Dict[str, float],
    ) -> Dict[str, TradingAction]:
        """Coordinate agent actions using weighted voting"""
        if not agent_actions:
            return {}

        # Calculate weighted votes for each action type
        action_votes = defaultdict(float)
        total_weight = 0

        for agent_name, action in agent_actions.items():
            weight = (
                self.agent_weights.get(agent_name, 1.0) * agent_confidences[agent_name]
            )
            action_votes[action.action_type] += weight
            total_weight += weight

        # Normalize votes
        if total_weight > 0:
            for action_type in action_votes:
                action_votes[action_type] /= total_weight

        # Select winning action
        winning_action = max(action_votes.items(), key=lambda x: x[1])
        action_type, confidence = winning_action

        # Calculate coordinated position size
        position_sizes = [
            action.position_size
            for action in agent_actions.values()
            if action.action_type == action_type
        ]
        avg_position_size = np.mean(position_sizes) if position_sizes else 0.0

        coordinated_action = TradingAction(
            action_type=action_type,
            position_size=avg_position_size,
            confidence=confidence,
            reasoning=f"Multi-agent coordination: {len(agent_actions)} agents",
        )

        return {"coordinated": coordinated_action}

    def update_agent_weights(self, performance_metrics: Dict[str, float]):
        """Update agent weights based on performance"""
        total_performance = sum(performance_metrics.values())

        if total_performance > 0:
            for agent_name, performance in performance_metrics.items():
                if agent_name in self.agent_weights:
                    new_weight = performance / total_performance
                    # Smooth update
                    self.agent_weights[agent_name] = (
                        0.7 * self.agent_weights[agent_name] + 0.3 * new_weight
                    )

        logger.info(f"Updated agent weights: {self.agent_weights}")

    def save_models(self, directory: str):
        """Save all agent models"""
        if not TORCH_AVAILABLE:
            return

        Path(directory).mkdir(parents=True, exist_ok=True)

        for agent_name, agent in self.agents.items():
            model_path = Path(directory) / f"{agent_name}_model.pth"
            torch.save(agent.state_dict(), model_path)

        # Save configuration
        config_path = Path(directory) / "rl_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        logger.info(f"Saved RL models to {directory}")

    def load_models(self, directory: str):
        """Load all agent models"""
        if not TORCH_AVAILABLE:
            return

        config_path = Path(directory) / "rl_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
                self.config = RLConfig(**config_dict)

        for agent_name in self.agents:
            model_path = Path(directory) / f"{agent_name}_model.pth"
            if model_path.exists():
                self.agents[agent_name].load_state_dict(torch.load(model_path))
                logger.info(f"Loaded model for agent: {agent_name}")


class DummyAgent:
    """Dummy agent for when PyTorch is not available"""

    def __init__(self):
        self.action_history = []

    def __call__(self, state):
        return np.random.random(3)

    def parameters(self):
        return []
