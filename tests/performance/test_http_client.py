"""
Test HTTP Client Optimization
Tests connection pooling, keep-alive connections, compression, retry logic, and performance improvements.
"""

import pytest
import time
import json
import threading
import os
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.http_client import HTTPClientManager, get_http_client_manager, optimized_get, optimized_post


class TestHTTPClientOptimization:
    """Test suite for HTTP client optimizations"""

    @pytest.fixture
    def http_client(self):
        """Get HTTP client manager instance"""
        return get_http_client_manager()

    def test_singleton_pattern(self):
        """Test that HTTPClientManager follows singleton pattern"""
        client1 = HTTPClientManager()
        client2 = HTTPClientManager()

        assert client1 is client2
        assert client1._instance is client2._instance

    def test_session_creation_and_reuse(self, http_client):
        """Test session creation and reuse"""
        # Get session
        session1 = http_client.get_session('test_session')
        assert session1 is not None

        # Get same session again (should reuse)
        session2 = http_client.get_session('test_session')
        assert session1 is session2

        # Different session name should create new session
        session3 = http_client.get_session('different_session')
        assert session3 is not session1

    def test_connection_pooling_config(self, http_client):
        """Test connection pooling configuration"""
        session = http_client.get_session()

        # Check that session has HTTPAdapter with pooling
        adapter = session.adapters.get('https://')
        assert adapter is not None
        assert adapter.config['pool_connections'] == http_client.config['pool_connections']
        assert adapter.config['pool_maxsize'] == http_client.config['pool_maxsize']

    @patch('requests.Session.request')
    def test_request_with_metrics(self, mock_request, http_client):
        """Test request execution with metrics tracking"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Encoding': 'gzip'}
        mock_request.return_value = mock_response

        # Make request
        initial_metrics = http_client.get_metrics()
        response = http_client.get('https://httpbin.org/get')

        # Check that request was made
        mock_request.assert_called_once()

        # Check metrics were updated
        final_metrics = http_client.get_metrics()
        assert final_metrics['requests_total'] == initial_metrics['requests_total'] + 1
        assert final_metrics['requests_success'] == initial_metrics['requests_success'] + 1
        assert final_metrics['compression_savings'] == initial_metrics['compression_savings'] + 1

    @patch('requests.Session.request')
    def test_retry_logic(self, mock_request, http_client):
        """Test retry logic on failures"""
        # Mock request to fail twice then succeed
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            mock_response
        ]

        # Make request
        response = http_client.get('https://httpbin.org/get')

        # Should have been called 3 times (initial + 2 retries)
        assert mock_request.call_count == 3

    def test_compression_headers(self, http_client):
        """Test compression headers are set correctly"""
        session = http_client.get_session()

        expected_headers = {
            'User-Agent': http_client.config['user_agent'],
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }

        for key, value in expected_headers.items():
            assert session.headers.get(key) == value

    @patch('requests.Session.request')
    def test_concurrent_requests(self, mock_request, http_client):
        """Test concurrent HTTP requests"""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_request.return_value = mock_response

        def make_request(url):
            """Worker function for concurrent requests"""
            return http_client.get(url)

        # Make concurrent requests
        urls = [f'https://api.example.com/endpoint{i}' for i in range(10)]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, url) for url in urls]
            results = [future.result() for future in as_completed(futures)]

        # All requests should have succeeded
        assert len(results) == 10
        assert all(result is not None for result in results)

        # Total requests should equal number of URLs
        assert mock_request.call_count == 10

    @patch('requests.Session.request')
    def test_performance_metrics_tracking(self, mock_request, http_client):
        """Test performance metrics are tracked correctly"""
        # Mock responses with different timings
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_request.return_value = mock_response

        # Reset metrics
        http_client.reset_metrics()
        initial_metrics = http_client.get_metrics()

        # Make multiple requests
        for i in range(5):
            http_client.get(f'https://api.example.com/test{i}')

        final_metrics = http_client.get_metrics()

        # Check metrics
        assert final_metrics['requests_total'] == initial_metrics['requests_total'] + 5
        assert final_metrics['requests_success'] == initial_metrics['requests_success'] + 5
        assert final_metrics['avg_response_time'] > 0

    @patch('requests.Session.request')
    def test_error_handling_and_metrics(self, mock_request, http_client):
        """Test error handling updates failure metrics"""
        # Mock failed request
        mock_request.side_effect = Exception("Network error")

        # Reset metrics
        http_client.reset_metrics()
        initial_metrics = http_client.get_metrics()

        # Make request that will fail
        with pytest.raises(Exception):
            http_client.get('https://api.example.com/failing-endpoint')

        final_metrics = http_client.get_metrics()

        # Check failure metrics
        assert final_metrics['requests_total'] == initial_metrics['requests_total'] + 1
        assert final_metrics['requests_failed'] == initial_metrics['requests_failed'] + 1

    def test_session_context_manager(self, http_client):
        """Test session context manager"""
        with http_client.session_context('test_context') as session:
            assert session is not None
            assert session is http_client.get_session('test_context')

    def test_convenience_functions(self):
        """Test convenience functions work correctly"""
        with patch('core.http_client.get_http_client_manager') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Test optimized_get
            optimized_get('https://example.com')
            mock_client.get.assert_called_once_with('https://example.com', 'default', timeout=mock_client.config['timeout'])

            # Test optimized_post
            mock_client.reset_mock()
            optimized_post('https://example.com', data={'key': 'value'})
            mock_client.post.assert_called_once()

    def test_backward_compatibility(self):
        """Test backward compatibility functions"""
        with patch('core.http_client.get_http_client_manager') as mock_get_client:
            mock_client = Mock()
            mock_response = Mock()
            mock_client.get.return_value = mock_response
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            from core.http_client import get_with_pool, post_with_pool

            # Test get_with_pool
            response = get_with_pool('https://example.com')
            assert response == mock_response

            # Test post_with_pool
            response = post_with_pool('https://example.com', data={'test': 'data'})
            assert response == mock_response

    @patch('requests.Session.request')
    def test_keep_alive_connections(self, mock_request, http_client):
        """Test keep-alive connection reuse"""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Connection': 'keep-alive'}
        mock_request.return_value = mock_response

        # Make multiple requests to same host
        for i in range(3):
            http_client.get('https://api.example.com/data')

        # Should reuse the same session/connection
        assert mock_request.call_count == 3

        # All calls should use the same session
        session_calls = [call[0][0] for call in mock_request.call_args_list]
        assert all(call == session_calls[0] for call in session_calls)

    def test_thread_safety(self, http_client):
        """Test thread-safe session management"""
        results = []
        errors = []

        def worker_thread(thread_id):
            """Worker function for thread safety test"""
            try:
                session = http_client.get_session(f'thread_{thread_id}')
                assert session is not None
                results.append(f'thread_{thread_id}_success')
            except Exception as e:
                errors.append(f'thread_{thread_id}_error: {e}')

        # Run multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker_thread, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check results
        assert len(results) == 10
        assert len(errors) == 0
        assert all('success' in result for result in results)

    def test_configuration_override(self):
        """Test configuration can be overridden via environment variables"""
        original_env = dict(os.environ)

        try:
            # Set environment variables
            os.environ['HTTP_POOL_CONNECTIONS'] = '10'
            os.environ['HTTP_MAX_RETRIES'] = '5'
            os.environ['HTTP_ENABLE_COMPRESSION'] = 'false'

            # Create new client instance (bypassing singleton for test)
            client = HTTPClientManager()
            client._instance = None  # Reset singleton
            client.__init__()

            # Check configuration was applied
            assert client.config['pool_connections'] == 10
            assert client.config['max_retries'] == 5
            assert client.config['enable_compression'] == False

        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)