#!/usr/bin/env python3
"""
Configuration Validation Module for ORACLE-X

This module provides comprehensive validation for environment variables
and configuration settings to prevent runtime errors and provide clear
error messages for missing or invalid configurations.
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class ConfigValidationLevel(Enum):
    """Configuration validation levels"""
    STRICT = "strict"      # All required configs must be present
    PERMISSIVE = "permissive"  # Allow missing configs with warnings
    MINIMAL = "minimal"    # Only validate critical configs


@dataclass
class ConfigValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missing_configs: List[str]
    invalid_configs: List[str]


class ConfigValidator:
    """Validates ORACLE-X configuration settings"""
    
    # Required configuration variables
    REQUIRED_CONFIGS = {
        'OPENAI_API_KEY': {
            'description': 'OpenAI API key for LLM operations',
            'validation': lambda x: x and len(x) > 10,
            'error_msg': 'Must be a valid OpenAI API key (sk-...)'
        }
    }
    
    # Optional but recommended configurations
    RECOMMENDED_CONFIGS = {
        'OPENAI_MODEL': {
            'description': 'Preferred OpenAI model',
            'default': 'gpt-4o-mini',
            'validation': lambda x: x in ['gpt-4o-mini', 'gpt-4o', 'o4-mini', 'o3-mini'],
            'error_msg': 'Should be a supported OpenAI model'
        },
        'QDRANT_URL': {
            'description': 'Qdrant vector database URL',
            'default': 'http://localhost:6333',
            'validation': lambda x: x.startswith(('http://', 'https://')),
            'error_msg': 'Must be a valid URL starting with http:// or https://'
        },
        'TWELVEDATA_API_KEY': {
            'description': 'TwelveData API key for market data',
            'validation': lambda x: x and len(x) > 5,
            'error_msg': 'Must be a valid TwelveData API key'
        }
    }
    
    # Optional configurations
    OPTIONAL_CONFIGS = {
        'OPENAI_API_BASE': {
            'description': 'Custom OpenAI API base URL',
            'default': 'https://api.openai.com/v1',
            'validation': lambda x: x.startswith(('http://', 'https://')),
            'error_msg': 'Must be a valid URL'
        },
        'FALLBACK_MODELS': {
            'description': 'Comma-separated fallback models',
            'validation': lambda x: ',' in x or x in ['gpt-4o-mini', 'gpt-4o'],
            'error_msg': 'Should be comma-separated model names'
        },
        'ALPHAVANTAGE_API_KEY': {
            'description': 'Alpha Vantage API key',
            'validation': lambda x: len(x) > 5,
            'error_msg': 'Must be a valid Alpha Vantage API key'
        },
        'FINNHUB_API_KEY': {
            'description': 'Finnhub API key',
            'validation': lambda x: len(x) > 5,
            'error_msg': 'Must be a valid Finnhub API key'
        },
        'REDDIT_CLIENT_ID': {
            'description': 'Reddit API client ID',
            'validation': lambda x: len(x) > 5,
            'error_msg': 'Must be a valid Reddit client ID'
        },
        'REDDIT_CLIENT_SECRET': {
            'description': 'Reddit API client secret',
            'validation': lambda x: len(x) > 5,
            'error_msg': 'Must be a valid Reddit client secret'
        }
    }
    
    @classmethod
    def validate_environment(cls, 
                           level: ConfigValidationLevel = ConfigValidationLevel.PERMISSIVE) -> ConfigValidationResult:
        """
        Validate all environment configurations
        
        Args:
            level: Validation strictness level
            
        Returns:
            ConfigValidationResult with validation details
        """
        errors = []
        warnings = []
        missing_configs = []
        invalid_configs = []
        
        # Check required configurations
        for config_name, config_info in cls.REQUIRED_CONFIGS.items():
            value = os.getenv(config_name)
            
            if not value:
                missing_configs.append(config_name)
                error_msg = f"Missing required config: {config_name} - {config_info['description']}"
                errors.append(error_msg)
            elif not config_info['validation'](value):
                invalid_configs.append(config_name)
                error_msg = f"Invalid config {config_name}: {config_info['error_msg']}"
                errors.append(error_msg)
        
        # Check recommended configurations
        for config_name, config_info in cls.RECOMMENDED_CONFIGS.items():
            value = os.getenv(config_name)
            
            if not value:
                if level == ConfigValidationLevel.STRICT:
                    missing_configs.append(config_name)
                    errors.append(f"Missing recommended config: {config_name} - {config_info['description']}")
                else:
                    warnings.append(f"Missing recommended config: {config_name} - using default: {config_info.get('default', 'None')}")
            elif 'validation' in config_info and not config_info['validation'](value):
                invalid_configs.append(config_name)
                if level == ConfigValidationLevel.STRICT:
                    errors.append(f"Invalid config {config_name}: {config_info['error_msg']}")
                else:
                    warnings.append(f"Invalid config {config_name}: {config_info['error_msg']}")
        
        # Check optional configurations (only validate if present)
        for config_name, config_info in cls.OPTIONAL_CONFIGS.items():
            value = os.getenv(config_name)
            
            if value and 'validation' in config_info and not config_info['validation'](value):
                invalid_configs.append(config_name)
                warnings.append(f"Invalid optional config {config_name}: {config_info['error_msg']}")
        
        is_valid = len(errors) == 0
        
        return ConfigValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            missing_configs=missing_configs,
            invalid_configs=invalid_configs
        )
    
    @classmethod
    def print_validation_report(cls, result: ConfigValidationResult) -> None:
        """Print a formatted validation report"""
        print("🔍 ORACLE-X Configuration Validation Report")
        print("=" * 50)
        
        if result.is_valid:
            print("✅ Configuration validation PASSED")
        else:
            print("❌ Configuration validation FAILED")
        
        if result.errors:
            print(f"\n🚨 Errors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  • {error}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  • {warning}")
        
        if result.missing_configs:
            print(f"\n📋 Missing Configurations ({len(result.missing_configs)}):")
            for config in result.missing_configs:
                print(f"  • {config}")
        
        if result.invalid_configs:
            print(f"\n❌ Invalid Configurations ({len(result.invalid_configs)}):")
            for config in result.invalid_configs:
                print(f"  • {config}")
        
        print("\n💡 Configuration Help:")
        print("  • Copy .env.example to .env and configure your API keys")
        print("  • See README.md for detailed configuration instructions")
        print("  • Use 'python config_validator.py --help' for more options")
    
    @classmethod
    def get_configuration_summary(cls) -> Dict[str, Any]:
        """Get a summary of current configuration status"""
        result = cls.validate_environment()
        
        summary = {
            'validation_status': 'PASSED' if result.is_valid else 'FAILED',
            'total_configs': len(cls.REQUIRED_CONFIGS) + len(cls.RECOMMENDED_CONFIGS) + len(cls.OPTIONAL_CONFIGS),
            'configured_count': 0,
            'missing_count': len(result.missing_configs),
            'invalid_count': len(result.invalid_configs),
            'warnings_count': len(result.warnings),
            'errors_count': len(result.errors),
            'configuration_completeness': 0.0
        }
        
        # Count configured items
        all_configs = {**cls.REQUIRED_CONFIGS, **cls.RECOMMENDED_CONFIGS, **cls.OPTIONAL_CONFIGS}
        configured_count = sum(1 for config in all_configs.keys() if os.getenv(config))
        
        summary['configured_count'] = configured_count
        summary['configuration_completeness'] = (configured_count / summary['total_configs']) * 100
        
        return summary


def validate_and_exit_on_error(level: ConfigValidationLevel = ConfigValidationLevel.PERMISSIVE) -> None:
    """
    Validate configuration and exit with error code if validation fails
    
    Args:
        level: Validation strictness level
    """
    result = ConfigValidator.validate_environment(level)
    ConfigValidator.print_validation_report(result)
    
    if not result.is_valid:
        print("\n💥 Exiting due to configuration errors. Please fix the above issues.")
        sys.exit(1)
    
    if result.warnings and level == ConfigValidationLevel.STRICT:
        print("\n⚠️  Warnings detected in strict mode. Please review configuration.")
        sys.exit(1)


def main():
    """CLI entry point for configuration validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate ORACLE-X configuration')
    parser.add_argument('--level', choices=['strict', 'permissive', 'minimal'], 
                       default='permissive', help='Validation level')
    parser.add_argument('--summary', action='store_true', 
                       help='Show configuration summary')
    parser.add_argument('--exit-on-error', action='store_true',
                       help='Exit with error code if validation fails')
    
    args = parser.parse_args()
    
    level = ConfigValidationLevel(args.level)
    
    if args.summary:
        summary = ConfigValidator.get_configuration_summary()
        print("📊 Configuration Summary:")
        print(f"  Status: {summary['validation_status']}")
        print(f"  Completeness: {summary['configuration_completeness']:.1f}%")
        print(f"  Configured: {summary['configured_count']}/{summary['total_configs']}")
        print(f"  Issues: {summary['errors_count']} errors, {summary['warnings_count']} warnings")
    
    if args.exit_on_error:
        validate_and_exit_on_error(level)
    else:
        result = ConfigValidator.validate_environment(level)
        ConfigValidator.print_validation_report(result)


if __name__ == '__main__':
    main()
