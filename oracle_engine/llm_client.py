"""
LLM Client - Backward compatibility wrapper over centralized llm_dispatcher.

All LLM calls now route through the centralized dispatcher with caching, retries, and unified provider handling.
"""

import logging
from typing import Any, Dict, List, Optional

from dataclasses import dataclass
from enum import Enum

from oracle_engine.dispatchers.llm_dispatcher import dispatch_chat, LLMDispatchResult
from oracle_engine.tools import get_sentiment, analyze_chart

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM providers for compatibility."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"


class ModelCapability(Enum):
    """Model capabilities for compatibility."""

    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    LONG_CONTEXT = "long_context"


@dataclass
class LLMRequest:
    """Request structure for compatibility."""

    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    model_preferences: Optional[List[ModelCapability]] = None
    cost_optimization: bool = True
    task_type: str = "general"


@dataclass
class LLMResponse:
    """Response structure for compatibility."""

    content: str
    model_used: str
    provider_used: str
    tokens_used: Dict[str, int]
    cost: float
    response_time: float
    success: bool
    error_message: Optional[str] = None


class LLMClient:
    """Unified LLM client wrapper over dispatcher."""

    def __init__(self):
        pass

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "auto",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        task_type: str = "general",
        cost_optimization: bool = True,
    ) -> LLMResponse:
        """Create chat completion using dispatcher."""
        result: LLMDispatchResult = dispatch_chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            task_type=task_type,
        )
        provider_str = result.provider or "openai"
        provider_enum = ProviderType.OPENAI  # default
        try:
            provider_enum = ProviderType[provider_str.upper()]
        except KeyError:
            pass
        return LLMResponse(
            content=result.content,
            model_used=result.model,
            provider_used=provider_enum.value,
            tokens_used=result.tokens_used or {"input_tokens": 0, "output_tokens": 0},
            cost=0.0,
            response_time=0.0,
            success=result.error is None,
            error_message=result.error,
        )

    def get_available_models(self) -> List[str]:
        """Compatibility stub."""
        return ["gpt-4o", "gpt-4o-mini", "claude-3-sonnet"]

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Compatibility stub."""
        return None

    def get_analytics(self) -> Dict[str, Any]:
        """Compatibility stub."""
        return {}

    def enable_provider(self, provider_type: str) -> bool:
        """Compatibility stub."""
        return True

    def disable_provider(self, provider_type: str) -> bool:
        """Compatibility stub."""
        return False


# Global client instance (wrapper)
_llm_client = LLMClient()


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance."""
    return _llm_client


def create_chat_completion(*args, **kwargs) -> LLMResponse:
    """Convenience function."""
    return _llm_client.create_chat_completion(*args, **kwargs)


# Compatibility layer for OpenAI
class ChatCompletions:
    """Drop-in replacement for OpenAI's ChatCompletions."""

    @staticmethod
    def create(*args, **kwargs) -> Any:
        """Create a chat completion."""
        response = _llm_client.create_chat_completion(*args, **kwargs)

        class Choice:
            def __init__(self, content):
                self.message = type("Message", (), {"content": content})()

        class Usage:
            def __init__(self, input_tokens, output_tokens):
                self.prompt_tokens = input_tokens
                self.completion_tokens = output_tokens
                self.total_tokens = input_tokens + output_tokens

        class OpenAIResponse:
            def __init__(self, response: LLMResponse):
                self.choices = [Choice(response.content)]
                self.usage = Usage(
                    response.tokens_used.get("input_tokens", 0),
                    response.tokens_used.get("output_tokens", 0),
                )
                self.model = response.model_used

        return OpenAIResponse(response)


class OpenAI:
    """Mock OpenAI client."""

    def __init__(self, api_key=None, base_url=None, timeout=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.chat = type("Chat", (), {"completions": ChatCompletions()})()


# Compatibility functions using tools/dispatcher
def get_sentiment(
    text: str, model_name: str = "auto", task_type: str = "analytical"
) -> str:
    """Sentiment analysis using dispatcher/tools."""
    return get_sentiment(text)


def analyze_chart(
    image_data: Optional[bytes], model_name: str = "auto", task_type: str = "analytical"
) -> str:
    """Chart analysis using dispatcher/tools."""
    return analyze_chart(image_data)


# Dummy provider manager for compatibility
class LLMProviderManager:
    def __init__(self):
        self.providers = {}
        self.models = {}

    def make_request_with_fallback(self, request: LLMRequest) -> LLMResponse:
        """Fallback to client."""
        return _llm_client.create_chat_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            task_type=request.task_type,
        )


def get_provider_manager() -> LLMProviderManager:
    return LLMProviderManager()
