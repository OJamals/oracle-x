# 🧪 API Integration Test Report
Generated: 2025-08-21 05:28:25

## 📊 Summary
- **Total Tests**: 5
- **Passed**: 5 ✅
- **Failed**: 0 ❌
- **Success Rate**: 100.0%

## 🔍 Detailed Results

### Qdrant Vector Database - ✅ PASS
**Message**: Qdrant fully accessible via client and HTTP
**Details**:
- collections_count: 1
- collections: ['qwen3_embedding']
- health_status: HTTP 404
- client_connection: Success
- http_health: False

### OpenAI/LLM Service - ✅ PASS
**Message**: LLM service working correctly
**Details**:
- model: gpt-4o
- response: API test successful
- usage: {'completion_tokens': 4, 'prompt_tokens': 28, 'total_tokens': 32, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': None, 'reasoning_tokens': None, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}

### Embedding Service - ✅ PASS
**Message**: Local embedding service accessible
**Details**:
- service_type: local
- models_available: 1
- models: ['qwen3-embedding']

### TwelveData API - ✅ PASS
**Message**: TwelveData API fully functional
**Details**:
- api_usage: {'timestamp': '2025-08-21 09:28:25', 'current_usage': 1, 'plan_limit': 8, 'daily_usage': 109, 'plan_daily_limit': 800, 'plan_category': 'basic'}
- test_price_call: {'price': '226.0099945'}
- endpoints_tested: ['api_usage', 'price']

### Reddit API - ✅ PASS
**Message**: Reddit API accessible
**Details**:
- test_posts_retrieved: 1
