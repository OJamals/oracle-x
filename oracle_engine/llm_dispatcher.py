"""
Unified LLM Dispatcher

Centralizes all LLM invocations with caching, retries, and config-driven models/providers.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from core.config import config
from oracle_engine.model_attempt_logger import log_attempt
from core.unified_cache_manager import UnifiedCacheManager

logger = logging.getLogger(__name__)

class LLMRequest:
    """Represents an LLM request"""
    def __init__(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7):
        self.model = model
        self.messages = messages
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __hash__(self):
        # Create hash for caching based on key components
        content = str(self.messages) + str(self.model) + str(self.max_tokens) + str(self.temperature)
        return hash(content)

    def __eq__(self, other):
        return (isinstance(other, LLMRequest) and
                self.model == other.model and
                self.messages == other.messages and
                self.max_tokens == other.max_tokens and
                self.temperature == other.temperature)

class LLMResponse:
    """Represents an LLM response"""
    def __init__(self, content: str, usage: Optional[Dict] = None, model: str = ""):
        self.content = content
        self.usage = usage or {}
        self.model = model

class LLMDispatcher:
    """Unified dispatcher for LLM calls with caching and retries"""

    def __init__(self):
        self.cache = UnifiedCacheManager()
        self.cache_ttl = config.cache.default_ttl_seconds if hasattr(config, 'cache') else 300
        self._init_clients()

    def _init_clients(self):
        """Initialize LLM clients"""
        self.clients = {}

        # OpenAI
        if config.model.openai_api_key:
            self.clients['openai'] = OpenAI(
                api_key=config.model.openai_api_key,
                base_url=config.model.openai_api_base or "https://api.openai.com/v1"
            )

        # Add other providers here as needed (Anthropic, etc.)

    def _get_client_for_model(self, model: str) -> Tuple[Optional[Any], str]:
        """Get appropriate client for model"""
        if model.startswith('gpt'):
            return self.clients.get('openai'), 'openai'
        # Add other model mappings
        return None, ''

    def _iter_fallback_models(self, primary: str) -> List[str]:
        """Return ordered list of models to try"""
        try:
            fallback = config.model.fallback_models
        except Exception:
            fallback = []
        ordered = [primary] + [m for m in fallback if m != primary]
        seen = set()
        result = []
        for m in ordered:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def call_llm(self, request: LLMRequest, use_cache: bool = True, retries: int = 3) -> LLMResponse:
        """
        Unified LLM call with caching and retries

        Args:
            request: LLMRequest object
            use_cache: Whether to use caching
            retries: Number of retries on failure

        Returns:
            LLMResponse object
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(str(hash(request)))
            if cached:
                logger.debug(f"Cache hit for LLM request")
                return LLMResponse(**cached)

        # Try models with fallbacks
        last_error = None
        for attempt, model in enumerate(self._iter_fallback_models(request.model)):
            if attempt >= retries:
                break

            client, provider = self._get_client_for_model(model)
            if not client:
                continue

            start_time = time.time()
            try:
                logger.debug(f"Trying model {model} (attempt {attempt + 1})")

                # Adjust request for provider
                if provider == 'openai':
                    response = client.chat.completions.create(
                        model=model,
                        messages=request.messages,
                        max_completion_tokens=request.max_tokens,
                        temperature=request.temperature
                    )
                    content = (response.choices[0].message.content or "").strip()
                    usage = {
                        'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                        'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                        'total_tokens': response.usage.total_tokens if response.usage else 0
                    }

                if content:
                    log_attempt("llm_dispatcher", model, start_time=start_time, success=True, empty=False, error=None)

                    response_obj = LLMResponse(content=content, usage=usage, model=model)

                    # Cache the result
                    if use_cache:
                        self.cache.set(str(hash(request)), {
                            'content': content,
                            'usage': usage,
                            'model': model
                        }, ttl=self.cache_ttl)

                    return response_obj
                else:
                    log_attempt("llm_dispatcher", model, start_time=start_time, success=False, empty=True, error=None)

            except Exception as e:
                error_msg = str(e)
                log_attempt("llm_dispatcher", model, start_time=start_time, success=False, empty=False, error=error_msg)
                logger.warning(f"Model {model} failed: {error_msg}")
                last_error = e
                continue

        # All models failed
        error_msg = f"All models failed. Last error: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def batch_call_llm(self, requests: List[LLMRequest], use_cache: bool = True, max_parallel: int = 5) -> List[LLMResponse]:
        """
        Batch LLM calls with parallel processing

        Args:
            requests: List of LLMRequest objects
            use_cache: Whether to use caching
            max_parallel: Maximum parallel calls

        Returns:
            List of LLMResponse objects
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        async def _batch_call():
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                tasks = [
                    loop.run_in_executor(executor, self.call_llm, req, use_cache, 3)
                    for req in requests
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)

        # Run async batch
        try:
            results = asyncio.run(_batch_call())
            # Handle exceptions
            responses = []
            for result in results:
                if isinstance(result, Exception):
                    # Return empty response on error
                    responses.append(LLMResponse(content="", model=""))
                else:
                    responses.append(result)
            return responses
        except Exception as e:
            logger.error(f"Batch call failed: {e}")
            return [LLMResponse(content="", model="") for _ in requests]

# Global instance
_dispatcher = None

def get_llm_dispatcher() -> LLMDispatcher:
    """Get global LLM dispatcher instance"""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = LLMDispatcher()
    return _dispatcher

# Convenience functions
def call_llm(model: str, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7,
             use_cache: bool = True, retries: int = 3) -> str:
    """Convenience function for LLM calls"""
    dispatcher = get_llm_dispatcher()
    request = LLMRequest(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    response = dispatcher.call_llm(request, use_cache=use_cache, retries=retries)
    return response.content

def batch_call_llm(requests_data: List[Dict], use_cache: bool = True, max_parallel: int = 5) -> List[str]:
    """Convenience function for batch LLM calls"""
    dispatcher = get_llm_dispatcher()
    requests = [
        LLMRequest(
            model=data['model'],
            messages=data['messages'],
            max_tokens=data.get('max_tokens', 1024),
            temperature=data.get('temperature', 0.7)
        )
        for data in requests_data
    ]
    responses = dispatcher.batch_call_llm(requests, use_cache=use_cache, max_parallel=max_parallel)
    return [r.content for r in responses]</content>
<parameter name="filePath">/Users/omar/Documents/Projects/oracle-x/oracle_engine/llm_dispatcher.py