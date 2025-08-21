"""
Multi-Model LLM Provider System for Oracle-X

This module provides a unified interface for multiple LLM providers including:
- OpenAI (GPT models)
- Anthropic (Claude models)
- Google (Gemini models)

Features:
- Provider failover and load balancing
- Intelligent model selection based on task requirements
- Cost optimization and rate limiting
- Quality comparison and analytics
- Unified interface for scenario generation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import threading
import json
import os
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ModelCapability(Enum):
    """Model capabilities for intelligent selection"""
    FAST = "fast"           # Quick responses, lower quality
    BALANCED = "balanced"   # Good balance of speed and quality
    HIGH_QUALITY = "high_quality"  # Best quality, slower
    CREATIVE = "creative"   # Good for creative tasks
    ANALYTICAL = "analytical"  # Good for analysis tasks
    LONG_CONTEXT = "long_context"  # Can handle long contexts


@dataclass
class ModelInfo:
    """Information about a specific model"""
    name: str
    provider: ProviderType
    capabilities: List[ModelCapability]
    cost_per_1k_input: float  # Cost in USD per 1K input tokens
    cost_per_1k_output: float  # Cost in USD per 1K output tokens
    max_tokens: int
    context_window: int
    is_available: bool = True


@dataclass
class ProviderConfig:
    """Configuration for a specific provider"""
    api_key: str
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    rate_limit_requests: int = 60  # requests per minute
    rate_limit_window: int = 60   # seconds


@dataclass
class LLMRequest:
    """Request structure for LLM calls"""
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    model_preferences: Optional[List[ModelCapability]] = None
    cost_optimization: bool = True
    task_type: str = "general"  # "creative", "analytical", "general", etc.


@dataclass
class LLMResponse:
    """Response structure from LLM calls"""
    content: str
    model_used: str
    provider_used: ProviderType
    tokens_used: Dict[str, int]  # input_tokens, output_tokens
    cost: float
    response_time: float
    success: bool
    error_message: Optional[str] = None


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    def __init__(self, config: ProviderConfig, models: List[ModelInfo]):
        self.config = config
        self.models = models
        self.request_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._rate_limit_tokens = 0
        self._rate_limit_window_start = time.time()

    @abstractmethod
    def _make_request(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        """Make actual request to the provider"""
        pass

    def make_request(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        """Make request with rate limiting and error handling"""
        self._enforce_rate_limit()

        start_time = time.time()
        try:
            response = self._make_request(request, model)
            response.response_time = time.time() - start_time
            self._log_request(response)
            return response
        except Exception as e:
            response = LLMResponse(
                content="",
                model_used=model.name,
                provider_used=model.provider,
                tokens_used={"input_tokens": 0, "output_tokens": 0},
                cost=0.0,
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            self._log_request(response)
            return response

    def _enforce_rate_limit(self):
        """Enforce rate limiting"""
        with self._lock:
            current_time = time.time()
            window_elapsed = current_time - self._rate_limit_window_start

            if window_elapsed >= self.config.rate_limit_window:
                # Reset window
                self._rate_limit_tokens = 0
                self._rate_limit_window_start = current_time
            elif self._rate_limit_tokens >= self.config.rate_limit_requests:
                # Wait for window to reset
                sleep_time = self.config.rate_limit_window - window_elapsed
                logger.info(f"Rate limit reached, sleeping {sleep_time}s")
                time.sleep(sleep_time)
                self._rate_limit_tokens = 0
                self._rate_limit_window_start = time.time()

            self._rate_limit_tokens += 1

    def _log_request(self, response: LLMResponse):
        """Log request for analytics"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": response.model_used,
            "provider": response.provider_used.value,
            "success": response.success,
            "response_time": response.response_time,
            "cost": response.cost,
            "tokens_used": response.tokens_used,
            "error_message": response.error_message
        }
        self.request_history.append(log_entry)

        # Keep only last 1000 requests for memory efficiency
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation"""

    def _make_request(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.openai.com/v1",
                timeout=self.config.timeout
            )

            max_tokens = request.max_tokens or min(4096, model.max_tokens)

            response = client.chat.completions.create(
                model=model.name,
                messages=request.messages,
                max_tokens=max_tokens,
                temperature=request.temperature
            )

            content = response.choices[0].message.content or ""

            # Estimate token usage if not provided
            input_tokens = response.usage.prompt_tokens if response.usage else len(str(request.messages)) // 4
            output_tokens = response.usage.completion_tokens if response.usage else len(content) // 4

            cost = (
                (input_tokens / 1000) * model.cost_per_1k_input +
                (output_tokens / 1000) * model.cost_per_1k_output
            )

            return LLMResponse(
                content=content,
                model_used=model.name,
                provider_used=ProviderType.OPENAI,
                tokens_used={"input_tokens": input_tokens, "output_tokens": output_tokens},
                cost=cost,
                response_time=0.0,  # Will be set by parent
                success=True
            )

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider implementation"""

    def _make_request(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )

            # Convert OpenAI format to Anthropic format
            system_message = ""
            anthropic_messages = []

            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            max_tokens = request.max_tokens or min(4096, model.max_tokens)

            response = client.messages.create(
                model=model.name,
                messages=anthropic_messages,
                system=system_message,
                max_tokens=max_tokens,
                temperature=request.temperature
            )

            content = response.content[0].text if response.content else ""

            # Estimate token usage
            input_tokens = len(str(request.messages)) // 4
            output_tokens = len(content) // 4

            cost = (
                (input_tokens / 1000) * model.cost_per_1k_input +
                (output_tokens / 1000) * model.cost_per_1k_output
            )

            return LLMResponse(
                content=content,
                model_used=model.name,
                provider_used=ProviderType.ANTHROPIC,
                tokens_used={"input_tokens": input_tokens, "output_tokens": output_tokens},
                cost=cost,
                response_time=0.0,  # Will be set by parent
                success=True
            )

        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")


class GoogleProvider(BaseLLMProvider):
    """Google provider implementation"""

    def _make_request(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.config.api_key)

            google_model = genai.GenerativeModel(model.name)

            # Convert messages to Google format
            prompt = ""
            for msg in request.messages:
                role = "User" if msg["role"] == "user" else "Model"
                prompt += f"{role}: {msg['content']}\n"

            max_tokens = request.max_tokens or min(4096, model.max_tokens)

            response = google_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=request.temperature
                )
            )

            content = response.text if response.text else ""

            # Estimate token usage
            input_tokens = len(prompt) // 4
            output_tokens = len(content) // 4

            cost = (
                (input_tokens / 1000) * model.cost_per_1k_input +
                (output_tokens / 1000) * model.cost_per_1k_output
            )

            return LLMResponse(
                content=content,
                model_used=model.name,
                provider_used=ProviderType.GOOGLE,
                tokens_used={"input_tokens": input_tokens, "output_tokens": output_tokens},
                cost=cost,
                response_time=0.0,  # Will be set by parent
                success=True
            )

        except Exception as e:
            raise Exception(f"Google API error: {str(e)}")


class LLMProviderManager:
    """Manages multiple LLM providers with failover and load balancing"""

    def __init__(self):
        self.providers: Dict[ProviderType, BaseLLMProvider] = {}
        self.models: Dict[str, ModelInfo] = {}
        self._lock = threading.Lock()

    def add_provider(self, provider_type: ProviderType, provider: BaseLLMProvider):
        """Add a provider to the manager"""
        self.providers[provider_type] = provider
        for model in provider.models:
            self.models[model.name] = model

    def get_available_models(self, capabilities: Optional[List[ModelCapability]] = None) -> List[ModelInfo]:
        """Get available models, optionally filtered by capabilities"""
        available_models = [model for model in self.models.values() if model.is_available]

        if capabilities:
            available_models = [
                model for model in available_models
                if any(cap in model.capabilities for cap in capabilities)
            ]

        return available_models

    def select_model(self, request: LLMRequest) -> ModelInfo:
        """Select the best model for the request based on preferences and availability"""
        available_models = self.get_available_models(request.model_preferences)

        if not available_models:
            # Fallback to any available model
            available_models = [model for model in self.models.values() if model.is_available]
            if not available_models:
                raise Exception("No models available")

        # Sort by cost (if optimization enabled) and quality
        if request.cost_optimization:
            available_models.sort(key=lambda x: x.cost_per_1k_output)

        # Prefer higher quality models for analytical tasks
        if request.task_type == "analytical":
            available_models.sort(key=lambda x: ModelCapability.HIGH_QUALITY not in x.capabilities)

        return available_models[0]

    def make_request_with_fallback(self, request: LLMRequest) -> LLMResponse:
        """Make request with automatic failover across providers"""
        tried_providers = set()
        last_error = None

        # Try preferred model first
        preferred_model = self.select_model(request)
        provider = self.providers[preferred_model.provider]

        if preferred_model.provider not in tried_providers:
            tried_providers.add(preferred_model.provider)
            response = provider.make_request(request, preferred_model)
            if response.success:
                return response
            last_error = response.error_message

        # Try other providers with similar capabilities
        available_models = self.get_available_models(request.model_preferences)
        for model in available_models:
            if model.provider in tried_providers:
                continue

            provider = self.providers[model.provider]
            tried_providers.add(model.provider)

            response = provider.make_request(request, model)
            if response.success:
                return response
            last_error = response.error_message

        # If all failed, return the last error
        return LLMResponse(
            content="",
            model_used=preferred_model.name,
            provider_used=preferred_model.provider,
            tokens_used={"input_tokens": 0, "output_tokens": 0},
            cost=0.0,
            response_time=0.0,
            success=False,
            error_message=f"All providers failed. Last error: {last_error}"
        )

    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics about provider performance"""
        analytics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "provider_stats": {},
            "model_stats": {}
        }

        for provider in self.providers.values():
            provider_stats = {
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "total_cost": 0.0,
                "average_response_time": 0.0
            }

            for log_entry in provider.request_history:
                analytics["total_requests"] += 1
                provider_stats["requests"] += 1

                if log_entry["success"]:
                    analytics["successful_requests"] += 1
                    provider_stats["successful"] += 1
                else:
                    analytics["failed_requests"] += 1
                    provider_stats["failed"] += 1

                analytics["total_cost"] += log_entry["cost"]
                provider_stats["total_cost"] += log_entry["cost"]

            if provider_stats["requests"] > 0:
                provider_stats["average_response_time"] = sum(
                    log_entry["response_time"] for log_entry in provider.request_history
                ) / provider_stats["requests"]

            analytics["provider_stats"][provider.__class__.__name__] = provider_stats

        if analytics["total_requests"] > 0:
            analytics["average_response_time"] = sum(
                log_entry["response_time"]
                for provider in self.providers.values()
                for log_entry in provider.request_history
            ) / analytics["total_requests"]

        return analytics


# Global provider manager instance
_provider_manager = LLMProviderManager()


def get_provider_manager() -> LLMProviderManager:
    """Get the global provider manager instance"""
    return _provider_manager


def initialize_providers():
    """Initialize all configured providers"""
    manager = get_provider_manager()

    # OpenAI Provider
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        openai_config = ProviderConfig(
            api_key=openai_key,
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )

        openai_models = [
            ModelInfo("gpt-4o", ProviderType.OPENAI, [ModelCapability.BALANCED, ModelCapability.ANALYTICAL], 0.005, 0.015, 4096, 128000),
            ModelInfo("gpt-4o-mini", ProviderType.OPENAI, [ModelCapability.FAST, ModelCapability.BALANCED], 0.00015, 0.0006, 4096, 128000),
            ModelInfo("gpt-4", ProviderType.OPENAI, [ModelCapability.HIGH_QUALITY, ModelCapability.ANALYTICAL], 0.03, 0.06, 4096, 8192),
            ModelInfo("gpt-3.5-turbo", ProviderType.OPENAI, [ModelCapability.FAST], 0.0005, 0.0015, 4096, 16385)
        ]

        openai_provider = OpenAIProvider(openai_config, openai_models)
        manager.add_provider(ProviderType.OPENAI, openai_provider)

    # Anthropic Provider
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        anthropic_config = ProviderConfig(
            api_key=anthropic_key,
            base_url="https://api.anthropic.com"
        )

        anthropic_models = [
            ModelInfo("claude-3-opus-20240229", ProviderType.ANTHROPIC, [ModelCapability.HIGH_QUALITY, ModelCapability.ANALYTICAL, ModelCapability.LONG_CONTEXT], 0.015, 0.075, 4096, 200000),
            ModelInfo("claude-3-sonnet-20240229", ProviderType.ANTHROPIC, [ModelCapability.BALANCED, ModelCapability.ANALYTICAL], 0.003, 0.015, 4096, 200000),
            ModelInfo("claude-3-haiku-20240307", ProviderType.ANTHROPIC, [ModelCapability.FAST, ModelCapability.BALANCED], 0.00025, 0.00125, 4096, 200000)
        ]

        anthropic_provider = AnthropicProvider(anthropic_config, anthropic_models)
        manager.add_provider(ProviderType.ANTHROPIC, anthropic_provider)

    # Google Provider
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        google_config = ProviderConfig(
            api_key=google_key,
            base_url="https://generativelanguage.googleapis.com"
        )

        google_models = [
            ModelInfo("gemini-1.5-pro", ProviderType.GOOGLE, [ModelCapability.HIGH_QUALITY, ModelCapability.ANALYTICAL, ModelCapability.LONG_CONTEXT], 0.00125, 0.005, 8192, 2097152),
            ModelInfo("gemini-1.5-flash", ProviderType.GOOGLE, [ModelCapability.FAST, ModelCapability.BALANCED], 0.000075, 0.0003, 8192, 1048576)
        ]

        google_provider = GoogleProvider(google_config, google_models)
        manager.add_provider(ProviderType.GOOGLE, google_provider)


# Initialize providers on module import
initialize_providers()