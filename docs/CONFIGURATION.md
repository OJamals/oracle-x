# Oracle-X Configuration Guide

## Overview
Oracle-X uses a centralized configuration system through `env_config.py` for consistency across all modules.

## Environment Variables

### Core API Configuration
- `OPENAI_API_KEY` - API key for OpenAI services (required)
- `OPENAI_API_BASE` - Base URL for OpenAI API (optional, defaults to OpenAI)
- `OPENAI_MODEL` - Primary chat/completion model (default: gpt-4o)
- `EMBEDDING_MODEL` - Embedding model (default: text-embedding-3-small)

### Data Sources
- `TWELVEDATA_API_KEY` - TwelveData API key for market data
- `RSS_FEEDS` - RSS feed URLs (comma-separated)
- `RSS_INCLUDE_ALL` - Include all RSS feeds (true/false)

### Optimization System
- `ORACLE_OPTIMIZATION_ENABLED` - Enable prompt optimization (true/false)
- See `optimization.env` for additional optimization parameters

### Social Media & Sentiment
- Reddit API credentials (see agent_bundle/README.md)
- Twitter API credentials (see agent_bundle/README.md)

## Configuration Files

### Primary Configuration
- `.env` - Main environment variables (copy from .env.example)
- `env_config.py` - Centralized configuration loader

### Specialized Configuration  
- `optimization.env` - Optimization system parameters
- `rss_feeds_config.env` - RSS feed configuration
- `optimization_config.json` - Detailed optimization parameters
- `config/data_feed_config.yaml` - Data feed adapter settings

## Usage Examples

### In Python Code
```python
import env_config

# Access configuration
api_key = env_config.OPENAI_API_KEY
model = env_config.OPENAI_MODEL
```

### Environment Setup
```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env

# Run with configuration
python main.py --mode standard
```

## Configuration Validation
The system automatically validates configuration on startup and provides helpful error messages for missing required variables.

## Security Notes
- Never commit actual API keys to version control
- Use `.env.example` files for documentation
- Keep sensitive configuration in `.env` files (ignored by git)
