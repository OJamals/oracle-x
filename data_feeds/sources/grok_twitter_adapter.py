"""
Grok-powered Twitter Sentiment Adapter using OpenRouter
Uses Grok 4.1 fast agent with tool calling to fetch and analyze Twitter posts.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import OpenAI

from data_feeds.data_feed_orchestrator import SentimentData
from data_feeds.twitter_feed import TwitterSentimentFeed

logger = logging.getLogger(__name__)

GROK_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

class GrokTwitterAdapter:
    """Grok agent-powered Twitter sentiment adapter"""
    
    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None):
        logger.info("Initializing GrokTwitterAdapter...")
        credentials = self._load_api_credentials(api_key, base_url)
        self.client = self._build_client(credentials)
        self.feed = TwitterSentimentFeed()
        self.source_name = "grok_twitter"

    def _load_api_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> Dict[str, str]:
        """Load API credentials from parameters or environment."""
        key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("GROK_API_KEY")
        if not key:
            raise RuntimeError("OpenRouter API key not configured. Set OPENROUTER_API_KEY or pass api_key.")

        url = base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
        return {"api_key": key, "base_url": url}

    def _build_client(self, credentials: Dict[str, str]) -> OpenAI:
        """Create an OpenAI client with consistent headers."""
        try:
            client = OpenAI(
                api_key=credentials["api_key"],
                base_url=credentials["base_url"],
                default_headers={
                    "HTTP-Referer": "https://oracle-x.local",
                    "X-Title": "Oracle-X Grok Twitter Agent"
                }
            )
            logger.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def fetch_tweets(self, query: str, limit: int = 20) -> str:
        """Tool function to fetch tweets"""
        try:
            tweets = self.feed.fetch(query, limit)
            texts = [{"text": tweet.get("text", ""), "tickers": tweet.get("tickers", [])} for tweet in tweets]
            return json.dumps(texts)
        except Exception as e:
            logger.error(f"Failed to fetch tweets: {e}")
            return json.dumps([])
    
    def _build_messages(self, symbol: str) -> List[Dict[str, str]]:
        """Prepare system and user prompts."""
        return [
            {
                "role": "system",
                "content": """You are an expert financial Twitter sentiment analyst.
Use the fetch_tweets tool to get recent relevant tweets about the stock symbol.
Then analyze the sentiment and respond ONLY with valid JSON:
{
  "sentiment": number between -1.0 (very bearish) and 1.0 (very bullish),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "brief explanation",
  "key_tweets": ["1-3 example tweets influencing the score"]
}
Do not respond with anything else."""
            },
            {
                "role": "user",
                "content": f"Analyze recent Twitter sentiment for stock symbol: {symbol}"
            }
        ]

    def _build_tools(self, default_limit: int) -> List[Dict[str, Any]]:
        """Define available tools for the Grok agent."""
        limit = max(1, min(default_limit, 50))
        return [
            {
                "type": "function",
                "function": {
                    "name": "fetch_tweets",
                    "description": "Fetch recent tweets matching a query. Use ticker symbol or '$TICKER sentiment'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query, e.g. '$AAPL' or 'AAPL stock'"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of tweets to fetch (1-50)",
                                "default": limit
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def _extract_json_payload(self, content: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse JSON content from model output."""
        if not content:
            return None
        cleaned = content.strip()
        for prefix in ("```json", "```"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                break
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    
    def _run_agent(self, symbol: str, limit: int) -> Dict[str, Any]:
        """Run Grok agent to analyze sentiment"""
        logger.info(f"Starting Grok agent for symbol: {symbol}")
        messages = self._build_messages(symbol)
        tools = self._build_tools(limit)
        max_iterations = 5

        for iteration in range(max_iterations):
            try:
                logger.info(f"Agent iteration {iteration + 1}/{max_iterations}: calling OpenAI API")
                response = self.client.chat.completions.create(
                    model=GROK_MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=500
                )
                logger.info(f"Agent iteration {iteration + 1}: API call successful")
                
                message = response.choices[0].message
                messages.append(message)
                
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "fetch_tweets":
                            args = json.loads(tool_call.function.arguments or "{}")
                            tweets_json = self.fetch_tweets(
                                args.get("query", symbol),
                                args.get("limit", limit)
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tweets_json
                            })
                else:
                    # Final response
                    parsed = self._extract_json_payload(message.content)
                    if parsed and "sentiment" in parsed:
                        logger.info(f"Agent completed successfully for {symbol}")
                        return parsed

                    logger.warning(f"Agent failed to return valid JSON for {symbol}")
                    return {"sentiment": 0.0, "confidence": 0.0, "reasoning": "Analysis failed"}
            
            except Exception as e:
                logger.error(f"Agent iteration {iteration + 1} failed: {e}")
                continue
        
        logger.warning(f"Agent loop exceeded max iterations for {symbol}")
        return {"sentiment": 0.0, "confidence": 0.0, "reasoning": "Agent loop exceeded max iterations"}
    
    def get_sentiment(self, symbol: str, limit: int = 20) -> Optional[SentimentData]:
        """Get Grok-powered Twitter sentiment"""
        try:
            agent_result = self._run_agent(symbol, limit)
            
            sentiment_data = SentimentData(
                symbol=symbol,
                sentiment_score=float(agent_result.get("sentiment", 0.0)),
                confidence=float(agent_result.get("confidence", 0.5)),
                source=self.source_name,
                timestamp=datetime.now(),
                sample_size=limit,
                raw_data={
                    "agent_result": agent_result,
                    "analysis_method": "grok_agent_tool_calling"
                }
            )
            
            logger.info(f"Grok Twitter sentiment for {symbol}: {agent_result.get('sentiment', 0):.3f} "
                       f"(conf: {agent_result.get('confidence', 0):.3f})")
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Grok Twitter adapter failed for {symbol}: {e}")
            return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Health status"""
        return {
            'source': self.source_name,
            'status': 'operational',
            'model': GROK_MODEL,
            'provider': 'openrouter',
            'base_url': getattr(self.client, "base_url", DEFAULT_OPENROUTER_BASE_URL),
            'twitter_feed_status': 'available',
            'agent_iterations_max': 5
        }

# Backward compatibility
GrokEnhancedTwitterAdapter = GrokTwitterAdapter

    def capabilities(self) -> set[str]:
        """SourceAdapterProtocol capabilities"""
        return {"sentiment"}

    def health(self) -> dict[str, Any]:
        """SourceAdapterProtocol health check"""
        return self.get_health_status()

    def fetch_sentiment(self, symbol: str, **kwargs) -> SentimentData:
        """SourceAdapterProtocol sentiment fetch"""
        limit = kwargs.get("limit", 20)
        return self.get_sentiment(symbol, limit)
