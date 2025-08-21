"""
LLM Client - Unified interface for multi-model LLM system

This module provides a drop-in replacement for the current OpenAI client usage
while leveraging the multi-model provider system for enhanced scenario generation.
"""

from typing import List, Dict, Any, Optional
import logging
import time

from .llm_providers import (
    get_provider_manager,
    LLMRequest,
    LLMResponse,
    ModelCapability,
    ProviderType
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client that works with multiple providers"""

    def __init__(self):
        self.provider_manager = get_provider_manager()

    def create_chat_completion(self, messages: List[Dict[str, str]],
                             model: str = "auto",
                             max_tokens: Optional[int] = None,
                             temperature: float = 0.7,
                             task_type: str = "general",
                             cost_optimization: bool = True) -> LLMResponse:
        """
        Create a chat completion using the multi-model system.

        This method provides a drop-in replacement for OpenAI's chat.completions.create()
        while leveraging multiple providers with failover and optimization.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name or "auto" for intelligent selection
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            task_type: Type of task for model selection
            cost_optimization: Whether to optimize for cost

        Returns:
            LLMResponse object with the completion
        """

        # Determine model preferences based on task type
        model_preferences = []
        if task_type == "analytical":
            model_preferences = [ModelCapability.ANALYTICAL, ModelCapability.HIGH_QUALITY]
        elif task_type == "creative":
            model_preferences = [ModelCapability.CREATIVE, ModelCapability.BALANCED]
        elif task_type == "fast":
            model_preferences = [ModelCapability.FAST]
        else:
            model_preferences = [ModelCapability.BALANCED]

        # Create request
        request = LLMRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model_preferences=model_preferences,
            cost_optimization=cost_optimization,
            task_type=task_type
        )

        # If specific model requested, try to use it first
        if model != "auto":
            specific_model = self.provider_manager.models.get(model)
            if specific_model and specific_model.is_available:
                provider = self.provider_manager.providers.get(specific_model.provider)
                if provider:
                    response = provider.make_request(request, specific_model)
                    if response.success:
                        return response

        # Use intelligent selection with fallback
        return self.provider_manager.make_request_with_fallback(request)

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.provider_manager.models.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        model = self.provider_manager.models.get(model_name)
        if not model:
            return None

        return {
            "name": model.name,
            "provider": model.provider.value,
            "capabilities": [cap.value for cap in model.capabilities],
            "cost_per_1k_input": model.cost_per_1k_input,
            "cost_per_1k_output": model.cost_per_1k_output,
            "max_tokens": model.max_tokens,
            "context_window": model.context_window,
            "is_available": model.is_available
        }

    def get_analytics(self) -> Dict[str, Any]:
        """Get performance analytics"""
        return self.provider_manager.get_analytics()

    def enable_provider(self, provider_type: str) -> bool:
        """Enable a specific provider"""
        try:
            provider_enum = ProviderType(provider_type)
            provider = self.provider_manager.providers.get(provider_enum)
            if provider:
                for model in provider.models:
                    model.is_available = True
                return True
        except ValueError:
            pass
        return False

    def disable_provider(self, provider_type: str) -> bool:
        """Disable a specific provider"""
        try:
            provider_enum = ProviderType(provider_type)
            provider = self.provider_manager.providers.get(provider_enum)
            if provider:
                for model in provider.models:
                    model.is_available = False
                return True
        except ValueError:
            pass
        return False


# Global client instance
_llm_client = LLMClient()


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance"""
    return _llm_client


def create_chat_completion(*args, **kwargs) -> LLMResponse:
    """
    Convenience function for creating chat completions.
    This provides a drop-in replacement for openai.ChatCompletion.create()
    """
    return _llm_client.create_chat_completion(*args, **kwargs)


# Compatibility layer for existing code
class ChatCompletions:
    """Drop-in replacement for OpenAI's ChatCompletions"""

    @staticmethod
    def create(*args, **kwargs) -> Any:
        """Create a chat completion"""
        response = _llm_client.create_chat_completion(*args, **kwargs)

        # Return in OpenAI-compatible format
        class Choice:
            def __init__(self, content):
                self.message = type('Message', (), {'content': content})()

        class Usage:
            def __init__(self, input_tokens, output_tokens):
                self.prompt_tokens = input_tokens
                self.completion_tokens = output_tokens
                self.total_tokens = input_tokens + output_tokens

        class OpenAIResponse:
            def __init__(self, response: LLMResponse):
                self.choices = [Choice(response.content)]
                self.usage = Usage(
                    response.tokens_used.get('input_tokens', 0),
                    response.tokens_used.get('output_tokens', 0)
                )
                self.model = response.model_used

        return OpenAIResponse(response)


# Mock OpenAI client for compatibility
class OpenAI:
    """Mock OpenAI client that uses the multi-model system"""

    def __init__(self, api_key=None, base_url=None, timeout=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.chat = type('Chat', (), {'completions': ChatCompletions()})()


def get_sentiment(text: str, model_name: str = "auto", task_type: str = "analytical") -> str:
    """
    Enhanced sentiment analysis using multi-model system
    """
    messages = [
        {"role": "system", "content": "Analyze the sentiment of the following text and provide a brief analysis."},
        {"role": "user", "content": text}
    ]

    response = _llm_client.create_chat_completion(
        messages=messages,
        model=model_name,
        max_tokens=300,
        task_type=task_type
    )

    return response.content if response.success else "Sentiment analysis failed"


def analyze_chart(image_data: Optional[bytes], model_name: str = "auto", task_type: str = "analytical") -> str:
    """
    Enhanced chart analysis using multi-model system
    """
    if not image_data:
        return "No chart data provided"

    # For now, we'll use text-based analysis since the multi-modal aspect
    # would require provider-specific implementations
    messages = [
        {"role": "system", "content": "Analyze the following chart data and provide insights."},
        {"role": "user", "content": f"Chart data length: {len(image_data)} bytes. Provide technical analysis."}
    ]

    response = _llm_client.create_chat_completion(
        messages=messages,
        model=model_name,
        max_tokens=500,
        task_type=task_type
    )

    return response.content if response.success else "Chart analysis failed"