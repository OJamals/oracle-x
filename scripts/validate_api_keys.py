#!/usr/bin/env python3
"""
🔑 API Key Configuration Validator
Validates that all required API keys are properly configured for production use.
This script should be run before deploying or starting production services.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_feeds.api_key_validator import validate_all_api_keys, print_api_key_report
from core.config import ConfigValidator, config, load_config

def main():
    """Main validation function"""
    print("🔍 Starting API Key Configuration Validation")
    print("=" * 60)
    
    # Load configuration
    load_config()

    # Check if we're running in the correct directory
    if not os.path.exists('.env.example'):
        print("❌ Error: Must run from project root directory")
        print("   Current directory:", os.getcwd())
        sys.exit(1)

    # Check if .env file exists
    env_file_exists = os.path.exists('.env')
    if not env_file_exists:
        print("⚠️  Warning: .env file not found")
        print("   Please copy .env.example to .env and configure your API keys")
        print("   Run: cp .env.example .env")

    print(f"\n📁 Environment file status: {'✅ Found' if env_file_exists else '❌ Missing'}")

    # Validate using the configuration manager
    print("\n🔧 Validating using Configuration Manager:")
    print("-" * 40)
    validator = ConfigValidator(config)
    config_validation = validator.validate()
    print(f"Valid: {config_validation.is_valid}")
    if config_validation.errors:
        print("Errors:")
        for error in config_validation.errors:
            print(f"  - {error}")
    if config_validation.warnings:
        print("Warnings:")
        for warning in config_validation.warnings:
            print(f"  - {warning}")

    # Also validate using the dedicated validator
    print("\n🔧 Validating using API Key Validator:")
    print("-" * 40)
    validator_validation = validate_all_api_keys()
    print_api_key_report()

    # Summary and recommendations
    summary = validator_validation['summary']
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    print(f"\n📊 Overall Status:")
    if summary['required_missing'] > 0:
        print(f"   🔴 CRITICAL: {summary['required_missing']} required API keys missing")
        print(f"   🚫 Production deployment NOT recommended")
    elif summary['missing'] > 0:
        print(f"   🟡 WARNING: {summary['missing']} optional API keys missing")
        print(f"   ⚠️  Some features may be limited")
    else:
        print(f"   ✅ SUCCESS: All API keys configured")
        print(f"   🚀 Ready for production deployment")

    # Specific recommendations
    if summary['required_missing'] > 0:
        print(f"\n🔧 REQUIRED ACTIONS:")
        missing_services = [
            service for service, validation in validator_validation['services'].items()
            if not validation['valid'] and validation.get('required', False)
        ]
        for service in missing_services:
            config = validator_validation['services'][service]
            env_var = config.get('env_var', 'UNKNOWN')
            print(f"   1. Set {env_var} in your .env file")
            print(f"   2. Obtain API key from {service} provider")
            print(f"   3. Test the configuration")

    if summary['missing'] > 0:
        print(f"\n💡 OPTIONAL IMPROVEMENTS:")
        print(f"   • Consider configuring optional API keys for enhanced functionality")
        print(f"   • Some features will work with fallback mechanisms")

    # Exit codes for CI/CD
    if summary['required_missing'] > 0:
        print(f"\n❌ VALIDATION FAILED - Required API keys missing")
        sys.exit(1)
    elif summary['missing'] > 0:
        print(f"\n⚠️  VALIDATION PASSED WITH WARNINGS - Optional keys missing")
        sys.exit(0)
    else:
        print(f"\n✅ VALIDATION PASSED - All systems ready")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)