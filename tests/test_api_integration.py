#!/usr/bin/env python3
"""
Comprehensive API Integration Test Suite
Tests all external services with proper configuration and diagnostics
"""

import os
import sys
import requests
import json
import time
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.config import config, load_config


class APIIntegrationTester:
    """Comprehensive API integration testing with detailed diagnostics."""

    def __init__(self):
        load_config()
        self.config = config
        self.results = {}
        self.print_config()

    def print_config(self):
        """Print current configuration for debugging."""
        print("🔧 Current API Configuration:")
        print(f"  OPENAI_API_BASE: {self.config.get('OPENAI_API_BASE')}")
        print(f"  EMBEDDING_API_BASE: {self.config.get('EMBEDDING_API_BASE')}")
        print(f"  QDRANT_URL: {self.config.get('QDRANT_URL')}")
        print(f"  OPENAI_MODEL: {self.config.get('OPENAI_MODEL')}")
        print(f"  EMBEDDING_MODEL: {self.config.get('EMBEDDING_MODEL')}")
        print()

    def test_qdrant_connectivity(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test Qdrant vector database connectivity."""
        print("🔍 Testing Qdrant Vector Database...")

        qdrant_url = self.config.get("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = self.config.get("QDRANT_API_KEY")

        # Test using the QdrantClient (proper method)
        try:
            from qdrant_client import QdrantClient

            print(f"  Testing Qdrant connection to: {qdrant_url}")

            # Initialize client with API key
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

            # Test basic connectivity
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]

            print("  ✅ Qdrant client connection successful")
            print(f"  Found {len(collection_names)} collections: {collection_names}")

            # Test basic health via HTTP with proper Bearer token format
            health_url = f"{qdrant_url}/health"
            headers = {}
            if qdrant_api_key:
                headers["Authorization"] = f"Bearer {qdrant_api_key}"

            health_response = requests.get(health_url, headers=headers, timeout=5)
            health_status = (
                "OK"
                if health_response.status_code == 200
                else f"HTTP {health_response.status_code}"
            )

            return (
                True,
                "Qdrant fully accessible via client and HTTP",
                {
                    "collections_count": len(collection_names),
                    "collections": collection_names,
                    "health_status": health_status,
                    "client_connection": "Success",
                    "http_health": health_response.status_code == 200,
                },
            )

        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Qdrant connection failed: {error_msg}")
            return False, f"Qdrant connection error: {error_msg}", {"error": error_msg}

    def test_openai_llm_service(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test OpenAI/GitHub Copilot LLM service."""
        print("🤖 Testing LLM Service (OpenAI/GitHub Copilot)...")

        api_base = self.config.get("OPENAI_API_BASE")
        api_key = os.environ.get("OPENAI_API_KEY")
        model = self.config.get("OPENAI_MODEL", "gpt-4o")

        if not api_base:
            return False, "OPENAI_API_BASE not configured", {}

        if not api_key:
            return False, "OPENAI_API_KEY not configured", {}

        try:
            # Test with OpenAI Python client for proper authentication
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=api_base)

            print(f"  Testing model: {model}")
            print(f"  API Base: {api_base}")

            # Make a simple test completion
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "Say 'API test successful' if you can respond.",
                    },
                ],
                max_tokens=10,
                temperature=0,
            )

            message = response.choices[0].message.content
            print(f"  ✅ LLM Response: {message}")

            return (
                True,
                "LLM service working correctly",
                {
                    "model": model,
                    "response": message,
                    "usage": (
                        response.usage.model_dump()
                        if hasattr(response, "usage") and response.usage
                        else {}
                    ),
                },
            )

        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ LLM test failed: {error_msg}")
            return False, f"LLM service error: {error_msg}", {"error": error_msg}

    def test_embedding_service(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test embedding service."""
        print("🔢 Testing Embedding Service...")

        embedding_base = self.config.get("EMBEDDING_API_BASE")
        api_key = os.environ.get("OPENAI_API_KEY")
        embedding_model = self.config.get("EMBEDDING_MODEL")

        if not embedding_base:
            return False, "EMBEDDING_API_BASE not configured", {}

        try:
            # Check if it's a local service or OpenAI service
            if "localhost" in embedding_base or "127.0.0.1" in embedding_base:
                # Test local embedding service
                print(f"  Testing local embedding service: {embedding_base}")

                # Try health endpoint first
                health_url = f"{embedding_base}/health"
                try:
                    health_response = requests.get(health_url, timeout=5)
                    if health_response.status_code == 200:
                        print("  ✅ Local embedding service health check passed")
                    else:
                        print(
                            f"  ⚠️  Health endpoint returned {health_response.status_code}"
                        )
                except:
                    print("  ⚠️  No health endpoint available")

                # Try models endpoint
                models_url = f"{embedding_base}/v1/models"
                models_response = requests.get(models_url, timeout=5)

                if models_response.status_code == 200:
                    models_data = models_response.json()
                    print(
                        f"  ✅ Models endpoint accessible. Available models: {len(models_data.get('data', []))}"
                    )

                    return (
                        True,
                        "Local embedding service accessible",
                        {
                            "service_type": "local",
                            "models_available": len(models_data.get("data", [])),
                            "models": [
                                model.get("id", "unknown")
                                for model in models_data.get("data", [])
                            ],
                        },
                    )
                else:
                    return (
                        False,
                        f"Models endpoint failed: {models_response.status_code}",
                        {"error": models_response.text},
                    )

            else:
                # Test OpenAI embedding service
                from openai import OpenAI

                client = OpenAI(api_key=api_key, base_url=embedding_base)

                print(f"  Testing OpenAI embedding with model: {embedding_model}")

                # Ensure we have a valid embedding model
                if not embedding_model:
                    embedding_model = "text-embedding-3-small"  # fallback

                # Test embedding creation
                response = client.embeddings.create(
                    model=embedding_model, input="test embedding"
                )

                embedding = response.data[0].embedding
                print(
                    f"  ✅ Embedding created successfully. Dimension: {len(embedding)}"
                )

                return (
                    True,
                    "OpenAI embedding service working",
                    {
                        "service_type": "openai",
                        "model": embedding_model,
                        "embedding_dimension": len(embedding),
                    },
                )

        except requests.exceptions.ConnectionError as e:
            return (
                False,
                f"Connection failed: {embedding_base} not accessible",
                {"error": str(e)},
            )
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Embedding test failed: {error_msg}")
            return False, f"Embedding service error: {error_msg}", {"error": error_msg}

    def test_reddit_api(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test Reddit API connectivity."""
        print("🔴 Testing Reddit API...")

        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        user_agent = os.environ.get("REDDIT_USER_AGENT")

        if not all([client_id, client_secret, user_agent]):
            return False, "Reddit API credentials not fully configured", {}

        try:
            # Test Reddit OAuth
            import praw

            reddit = praw.Reddit(
                client_id=client_id, client_secret=client_secret, user_agent=user_agent
            )

            # Test read-only access
            subreddit = reddit.subreddit("test")
            posts = list(subreddit.hot(limit=1))

            print(f"  ✅ Reddit API working. Retrieved {len(posts)} test posts")

            return True, "Reddit API accessible", {"test_posts_retrieved": len(posts)}

        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Reddit test failed: {error_msg}")
            return False, f"Reddit API error: {error_msg}", {"error": error_msg}

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all API integration tests."""
        print("🚀 Starting Comprehensive API Integration Test Suite")
        print("=" * 60)

        tests = [
            ("Qdrant Vector Database", self.test_qdrant_connectivity),
            ("OpenAI/LLM Service", self.test_openai_llm_service),
            ("Embedding Service", self.test_embedding_service),
            ("Reddit API", self.test_reddit_api),
        ]

        self.results = {}
        passed = 0
        failed = 0

        for test_name, test_func in tests:
            print(f"\n📊 {test_name}")
            print("-" * 40)

            try:
                success, message, details = test_func()
                self.results[test_name] = {
                    "success": success,
                    "message": message,
                    "details": details,
                    "timestamp": time.time(),
                }

                if success:
                    passed += 1
                    print(f"  ✅ PASSED: {message}")
                else:
                    failed += 1
                    print(f"  ❌ FAILED: {message}")

            except Exception as e:
                failed += 1
                error_msg = f"Test execution error: {str(e)}"
                print(f"  💥 ERROR: {error_msg}")
                self.results[test_name] = {
                    "success": False,
                    "message": error_msg,
                    "details": {"exception": str(e)},
                    "timestamp": time.time(),
                }

        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")

        return {
            "summary": {
                "passed": passed,
                "failed": failed,
                "success_rate": (
                    passed / (passed + failed) * 100 if (passed + failed) > 0 else 0
                ),
            },
            "results": self.results,
            "timestamp": time.time(),
        }

    def generate_report(self) -> str:
        """Generate a detailed test report."""
        if not self.results:
            return "No test results available. Run tests first."

        report = []
        report.append("# 🧪 API Integration Test Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        passed = sum(1 for r in self.results.values() if r["success"])
        failed = len(self.results) - passed
        success_rate = (passed / len(self.results) * 100) if self.results else 0

        report.append("## 📊 Summary")
        report.append(f"- **Total Tests**: {len(self.results)}")
        report.append(f"- **Passed**: {passed} ✅")
        report.append(f"- **Failed**: {failed} ❌")
        report.append(f"- **Success Rate**: {success_rate:.1f}%")
        report.append("")

        # Detailed results
        report.append("## 🔍 Detailed Results")
        report.append("")

        for test_name, result in self.results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            report.append(f"### {test_name} - {status}")
            report.append(f"**Message**: {result['message']}")

            if result["details"]:
                report.append("**Details**:")
                for key, value in result["details"].items():
                    report.append(f"- {key}: {value}")

            report.append("")

        return "\n".join(report)


def main():
    """Main execution function."""
    tester = APIIntegrationTester()

    # Run comprehensive tests
    results = tester.run_comprehensive_test()

    # Generate and save report
    report = tester.generate_report()

    # Save report to file
    report_file = "api_integration_test_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\n📄 Detailed report saved to: {report_file}")

    # Save results as JSON
    results_file = "api_integration_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"📊 Test results saved to: {results_file}")

    return results["summary"]["success_rate"] >= 80  # Return True if 80%+ tests pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
