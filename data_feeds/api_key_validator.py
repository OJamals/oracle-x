"""
🔑 API Key Validation and Fallback Management
Validates API keys and provides graceful fallback mechanisms when keys are missing or invalid.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class APIKeyValidator:
    """Validates API keys and manages fallback strategies"""

    # Define API key requirements and fallback behaviors
    API_KEY_CONFIG = {
        "financial_modeling_prep": {
            "env_var": "FINANCIALMODELINGPREP_API_KEY",
            "required": True,
            "fallback_available": True,
            "fallback_description": "Basic financial data only",
        },
        "finnhub": {
            "env_var": "FINNHUB_API_KEY",
            "required": True,
            "fallback_available": True,
            "fallback_description": "Rate-limited free tier data",
        },
        "alphavantage": {
            "env_var": "ALPHAVANTAGE_API_KEY",
            "required": True,
            "fallback_available": True,
            "fallback_description": "Limited daily requests",
        },
        "twitter_bearer": {
            "env_var": "TWITTER_BEARER_TOKEN",
            "required": False,
            "fallback_available": True,
            "fallback_description": "Twitter scraping via twscrape (no API key needed)",
        },
        "reddit_client_id": {
            "env_var": "REDDIT_CLIENT_ID",
            "required": False,
            "fallback_available": True,
            "fallback_description": "No social media sentiment",
        },
        "reddit_client_secret": {
            "env_var": "REDDIT_CLIENT_SECRET",
            "required": False,
            "fallback_available": True,
            "fallback_description": "No social media sentiment",
        },
    }

    def __init__(self):
        self._validation_cache = {}
        self._cache_ttl = timedelta(minutes=5)

    def validate_api_key(self, service_name: str) -> Dict[str, Any]:
        """Validate a specific API key"""
        if service_name not in self.API_KEY_CONFIG:
            return {
                "valid": False,
                "reason": f"Unknown service: {service_name}",
                "fallback_available": False,
            }

        config = self.API_KEY_CONFIG[service_name]
        env_var = config["env_var"]

        # Try to get API key from configuration manager first, then fallback to environment
        api_key = None
        try:
            from core.config import config

            data_config = config.data_feeds

            # Map service names to configuration attributes
            service_to_attr = {
                "financial_modeling_prep": "financial_modeling_prep_api_key",
                "finnhub": "finnhub_api_key",
                "alphavantage": "alphavantage_api_key",
                "twitter_bearer": "twitter_bearer_token",
                "reddit_client_id": "reddit_client_id",
                "reddit_client_secret": "reddit_client_secret",
            }

            if service_name in service_to_attr:
                attr_name = service_to_attr[service_name]
                api_key = getattr(data_config, attr_name, None)

        except ImportError:
            # Fallback to environment variables if config manager is not available
            api_key = os.getenv(env_var)

        # Check if API key exists and is not empty
        if not api_key or not api_key.strip():
            return {
                "valid": False,
                "reason": f"API key not found in environment variable {env_var}",
                "fallback_available": config["fallback_available"],
                "fallback_description": config["fallback_description"],
                "required": config["required"],
            }

        # Basic format validation (could be extended for specific services)
        if len(api_key.strip()) < 10:
            return {
                "valid": False,
                "reason": f"API key appears to be too short or invalid",
                "fallback_available": config["fallback_available"],
                "fallback_description": config["fallback_description"],
                "required": config["required"],
            }

        return {
            "valid": True,
            "service": service_name,
            "env_var": env_var,
            "required": config["required"],
            "fallback_available": config["fallback_available"],
        }

    def validate_all_api_keys(self) -> Dict[str, Any]:
        """Validate all configured API keys"""
        results = {
            "timestamp": datetime.now(),
            "services": {},
            "summary": {
                "total": len(self.API_KEY_CONFIG),
                "valid": 0,
                "invalid": 0,
                "missing": 0,
                "required_missing": 0,
            },
        }

        for service_name in self.API_KEY_CONFIG:
            validation = self.validate_api_key(service_name)
            results["services"][service_name] = validation

            if validation["valid"]:
                results["summary"]["valid"] += 1
            else:
                results["summary"]["invalid"] += 1
                if "not found" in validation.get("reason", ""):
                    results["summary"]["missing"] += 1
                    if validation.get("required", False):
                        results["summary"]["required_missing"] += 1

        return results

    def get_fallback_strategy(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get fallback strategy for a service when API key is invalid"""
        validation = self.validate_api_key(service_name)

        if validation["valid"]:
            return None  # No fallback needed

        if not validation.get("fallback_available", False):
            return None  # No fallback available

        return {
            "service": service_name,
            "reason": validation.get("reason", "API key invalid"),
            "fallback_available": True,
            "fallback_description": validation.get("fallback_description", ""),
            "required": validation.get("required", False),
        }

    def execute_with_fallback(
        self,
        service_name: str,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a function with automatic fallback if API key is invalid"""
        validation = self.validate_api_key(service_name)

        if validation["valid"]:
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function failed for {service_name}: {e}")
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                else:
                    raise

        # API key is invalid, use fallback if available
        if fallback_func and validation.get("fallback_available", False):
            logger.info(
                f"Using fallback for {service_name}: {validation.get('fallback_description', '')}"
            )
            return fallback_func(*args, **kwargs)
        else:
            raise ValueError(
                f"API key required for {service_name}: {validation.get('reason', 'Invalid API key')}"
            )

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of API keys"""
        validation = self.validate_all_api_keys()
        summary = validation["summary"]

        # Determine overall health
        if summary["required_missing"] > 0:
            health = "critical"
            message = f"{summary['required_missing']} required API keys missing"
        elif summary["missing"] > 0:
            health = "warning"
            message = f"{summary['missing']} API keys missing"
        elif summary["valid"] == summary["total"]:
            health = "healthy"
            message = "All API keys configured"
        else:
            health = "warning"
            message = "Some API keys invalid"

        return {"health": health, "message": message, "validation": validation}

    def print_validation_report(self) -> None:
        """Print a formatted validation report"""
        validation = self.validate_all_api_keys()
        summary = validation["summary"]

        print("\n" + "=" * 70)
        print("🔑 API KEY VALIDATION REPORT")
        print("=" * 70)

        print(f"\n📊 SUMMARY:")
        print(f"   Total Services: {summary['total']}")
        print(f"   Valid Keys: {summary['valid']}")
        print(f"   Missing Keys: {summary['missing']}")
        print(f"   Required Missing: {summary['required_missing']}")

        # Valid keys
        valid_services = [s for s, v in validation["services"].items() if v["valid"]]
        if valid_services:
            print(f"\n✅ VALID API KEYS:")
            for service in valid_services:
                config = self.API_KEY_CONFIG[service]
                print(f"   ✓ {service} ({config['env_var']})")

        # Missing keys
        missing_services = [
            (s, v) for s, v in validation["services"].items() if not v["valid"]
        ]
        if missing_services:
            print(f"\n❌ MISSING/INVALID API KEYS:")
            for service, validation in missing_services:
                config = self.API_KEY_CONFIG[service]
                required_mark = "🔴" if config["required"] else "🟡"
                print(f"   {required_mark} {service} ({config['env_var']})")
                print(f"      Reason: {validation.get('reason', 'Unknown')}")
                if validation.get("fallback_available"):
                    print(
                        f"      Fallback: {validation.get('fallback_description', 'Available')}"
                    )

        print("\n" + "=" * 70)


# Global validator instance
_api_key_validator = APIKeyValidator()


# Convenience functions
def validate_api_key(service_name: str) -> Dict[str, Any]:
    """Validate a specific API key"""
    return _api_key_validator.validate_api_key(service_name)


def validate_all_api_keys() -> Dict[str, Any]:
    """Validate all configured API keys"""
    return _api_key_validator.validate_all_api_keys()


def execute_with_fallback(
    service_name: str,
    primary_func: Callable,
    fallback_func: Optional[Callable] = None,
    *args,
    **kwargs,
) -> Any:
    """Execute a function with automatic fallback if API key is invalid"""
    return _api_key_validator.execute_with_fallback(
        service_name, primary_func, fallback_func, *args, **kwargs
    )


def get_api_key_health_status() -> Dict[str, Any]:
    """Get overall health status of API keys"""
    return _api_key_validator.get_health_status()


def print_api_key_report() -> None:
    """Print a formatted API key validation report"""
    _api_key_validator.print_validation_report()


if __name__ == "__main__":
    # Test the validator
    print_api_key_report()
