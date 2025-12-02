"""
Config Loader for Data Feeds
Provides a unified, typed configuration with layered overrides:
1) .env (if present) - loads environment variables for credentials
2) YAML (if present) - base config overrides
3) Production JSON (if present) - final overlay

If no config files are present, sensible defaults mirroring current orchestrator behavior are used.

Defaults (aligned with data_feeds.data_feed_orchestrator.SmartCache and RateLimiter):
- cache_ttls:
    quote: 30
    market_data_1d: 3600
    market_data_1h: 300
    sentiment: 600
    news: 1800
    company_info: 86400
    market_breadth: 300
    group_performance: 1800
- rate_limits (per-minute or per-window caps used in RateLimiter):
    FINNHUB: per_minute=60
    IEX_CLOUD: per_minute=100
    REDDIT: per_minute=60
    TWITTER: per_15min=100 (window 900s)
    FRED: per_minute=120
    FINVIZ: per_minute=12
- quotas (daily):
    FINNHUB: per_day=1000
    IEX_CLOUD: per_day=50000

This module intentionally does not add new runtime behavior; it only aggregates configuration.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Load .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv is optional; proceed if unavailable
    pass

logger = logging.getLogger(__name__)


@dataclass
class Config:
    enabled_sources: Dict[str, bool]
    priorities: Dict[str, List[str]]
    cache_ttls: Dict[str, int]
    rate_limits: Dict[str, Dict[str, int]]
    quotas: Dict[str, Dict[str, int]]
    credentials: Dict[str, str]
    options: Dict[str, Any]


def _safe_load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        logger.warning("PyYAML not installed; skipping YAML config load for %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                logger.warning("YAML at %s did not parse into a dict; ignoring", path)
                return {}
            return data
    except Exception as e:
        logger.warning("Failed to load YAML %s: %s", path, e)
        return {}


def _safe_load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
            if not isinstance(data, dict):
                logger.warning("JSON at %s did not parse into a dict; ignoring", path)
                return {}
            return data
    except Exception as e:
        logger.warning("Failed to load JSON %s: %s", path, e)
        return {}


def _defaults() -> Config:
    # SmartCache defaults (from data_feed_orchestrator.SmartCache)
    default_cache = {
        "quote": 30,
        "market_data_1d": 3600,
        "market_data_1h": 300,
        "sentiment": 600,
        "news": 1800,
        "company_info": 86400,
        "market_breadth": 300,
        "group_performance": 1800,
    }

    # RateLimiter defaults (from data_feed_orchestrator.RateLimiter)
    default_rate_limits = {
        "FINNHUB": {"per_minute": 60},
        "IEX_CLOUD": {"per_minute": 100},
        "REDDIT": {"per_minute": 60},
        # Twitter uses 100 per 15 minutes; represent both granularly and per_minute proxy
        "TWITTER": {"per_15min": 100, "per_minute": 7},
        "FRED": {"per_minute": 120},
        "FINVIZ": {"per_minute": 12},
    }

    # Daily quotas (from data_feed_orchestrator.RateLimiter)
    default_quotas = {
        "FINNHUB": {"per_day": 1000},
        "IEX_CLOUD": {"per_day": 50000},
    }

    # Enabled sources defaults: enable those present in orchestrator adapters
    default_enabled = {
        "yfinance": True,
        "finnhub": True,
        "iex_cloud": True,
        "reddit": True,
        "twitter": True,
        "fred": True,
        "finviz": True,
        "yahoo_news": True,
        "google_trends": True,
        "sec_edgar": True,
    }

    # Priorities: conservative defaults; YAML may override with more detailed maps
    default_priorities = {
        "quote": ["yfinance"],
        "historical": ["yfinance"],
        "company_info": ["yfinance"],
        "market_data_intraday": ["yfinance"],
        "market_data_daily": ["yfinance"],
        "market_breadth": ["finviz"],
        "group_performance": ["finviz"],
        # keep simple default for orchestrator paths
    }

    # Credentials from environment
    creds = {
        "FINNHUB_API_KEY": os.getenv("FINNHUB_API_KEY", ""),
        "FINANCIALMODELINGPREP_API_KEY": os.getenv("FINANCIALMODELINGPREP_API_KEY", ""),
        "IEX_CLOUD_API_KEY": os.getenv("IEX_CLOUD_API_KEY", ""),
        "ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
        "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID", ""),
        "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET", ""),
        "TWITTER_BEARER_TOKEN": os.getenv("TWITTER_BEARER_TOKEN", ""),
    }

    options: Dict[str, Any] = {}

    return Config(
        enabled_sources=default_enabled,
        priorities=default_priorities,
        cache_ttls=default_cache,
        rate_limits=default_rate_limits,
        quotas=default_quotas,
        credentials=creds,
        options=options,
    )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    env_path: Optional[str] = None,
    yaml_path: str = "config/data_feed_config.yaml",
    prod_json_path: str = "config/ml_production.json",
) -> Config:
    """
    Load the unified configuration:

    1) Load .env (explicit path if provided, else default behavior)
    2) Load YAML config and overlay onto defaults
    3) Load production JSON and overlay final keys (only keys matching Config fields are applied)

    Missing files are ignored. If PyYAML is unavailable, YAML is skipped with a warning.
    """
    # Optional explicit .env path
    if env_path and os.path.exists(env_path):
        try:
            from dotenv import load_dotenv as _load

            _load(env_path, override=True)
        except Exception:
            # best effort
            pass

    defaults = _defaults()

    # Map defaults to dict for merging
    merged: Dict[str, Any] = {
        "enabled_sources": dict(defaults.enabled_sources),
        "priorities": dict(defaults.priorities),
        "cache_ttls": dict(defaults.cache_ttls),
        "rate_limits": dict(defaults.rate_limits),
        "quotas": dict(defaults.quotas),
        "credentials": dict(defaults.credentials),
        "options": dict(defaults.options),
    }

    # YAML overlay (schema is flexible; map fields if present)
    yaml_data = _safe_load_yaml(yaml_path)
    if yaml_data:
        # Priorities: accept both "data_sources" or "priority" maps
        priorities: Dict[str, List[str]] = dict(merged.get("priorities", {}))
        if isinstance(yaml_data.get("data_sources"), dict):
            # Remap to our expected keys where sensible
            ds = yaml_data["data_sources"]
            if isinstance(ds.get("quote"), list):
                priorities["quote"] = [str(x) for x in ds["quote"]]
            if isinstance(ds.get("historical"), list):
                priorities["historical"] = [str(x) for x in ds["historical"]]
            if isinstance(ds.get("company_info"), list):
                priorities["company_info"] = [str(x) for x in ds["company_info"]]
        if isinstance(yaml_data.get("priority"), dict):
            pr = yaml_data["priority"]
            for k, v in pr.items():
                if isinstance(v, list):
                    priorities[str(k)] = [str(x) for x in v]
        merged["priorities"] = priorities

        # Cache TTLs: accept "cache_ttl" as flat seconds or nested maps; flatten to ints where possible
        if isinstance(yaml_data.get("cache_ttl"), dict):
            cache_ttls = dict(merged.get("cache_ttls", {}))
            for k, v in yaml_data["cache_ttl"].items():
                if isinstance(v, int):
                    cache_ttls[str(k)] = v
                elif isinstance(v, dict):
                    # Support structures like {"rth_seconds": 120, "off_hours_seconds": 1800}
                    # Use rth_seconds as base TTL if present; else off_hours_seconds; else ignore
                    ttl = v.get("rth_seconds") or v.get("off_hours_seconds")
                    if isinstance(ttl, int):
                        cache_ttls[str(k)] = ttl
                # else ignore unsupported types
            merged["cache_ttls"] = cache_ttls

        # Rate limits: remap various schemas to a simple per-source dict[int]
        if isinstance(yaml_data.get("rate_limits"), dict):
            rate_limits = dict(merged.get("rate_limits", {}))
            for provider, limits in yaml_data["rate_limits"].items():
                if isinstance(limits, dict):
                    # Support either calls/period or direct rpm style. Normalize to per_minute if possible.
                    if "calls" in limits and "period" in limits:
                        calls = limits.get("calls")
                        period = limits.get("period")
                        if (
                            isinstance(calls, int)
                            and isinstance(period, int)
                            and period > 0
                        ):
                            per_minute = int(calls * 60 / period)
                            rate_limits[str(provider).upper()] = {
                                "per_minute": per_minute
                            }
                    else:
                        # granular; try rpm first
                        if "rpm" in limits and isinstance(limits["rpm"], int):
                            rate_limits[str(provider).upper()] = {
                                "per_minute": int(limits["rpm"])
                            }
                        elif "rps" in limits and isinstance(limits["rps"], int):
                            rate_limits[str(provider).upper()] = {
                                "per_minute": int(limits["rps"] * 60)
                            }
                        # keep existing if not mappable
            merged["rate_limits"] = rate_limits

    # Production JSON overlay: only apply recognized top-level fields if present
    prod_data = _safe_load_json(prod_json_path)
    if prod_data:
        # As production JSON in this repo is ML-focused, we conservatively only allow "options" overlay
        if isinstance(prod_data, dict):
            options = dict(merged.get("options", {}))
            options = _deep_merge(options, prod_data)
            merged["options"] = options

    # Credentials finalization: allow env vars to override or fill
    cred_keys = [
        "FINNHUB_API_KEY",
        "FINANCIALMODELINGPREP_API_KEY",
        "IEX_CLOUD_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "TWITTER_BEARER_TOKEN",
    ]
    credentials = dict(merged.get("credentials", {}))
    for k in cred_keys:
        envv = os.getenv(k)
        if envv:
            credentials[k] = envv
    merged["credentials"] = credentials

    # Build typed Config
    return Config(
        enabled_sources=merged.get("enabled_sources", {}),
        priorities=merged.get("priorities", {}),
        cache_ttls=merged.get("cache_ttls", {}),
        rate_limits=merged.get("rate_limits", {}),
        quotas=merged.get("quotas", {}),
        credentials=merged.get("credentials", {}),
        options=merged.get("options", {}),
    )
