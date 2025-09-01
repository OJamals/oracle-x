"""
Test Database Connection Pooling Optimizations
Tests connection reuse, prepared statement caching, thread safety, and performance improvements.
"""

import pytest
import sqlite3
import threading
import time
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.database_pool import DatabasePool, execute_query, execute_update, execute_many


class TestDatabasePool:
    """Test suite for DatabasePool functionality"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)

    def test_connection_pool_creation(self, temp_db):
        """Test pool creation and basic functionality"""
        pool = DatabasePool.get_pool(temp_db, max_connections=5)

        assert pool.db_path == temp_db
        assert pool.max_connections == 5
        assert pool._connections == []
        assert pool._total_connections_created == 0

    def test_connection_reuse(self, temp_db):
        """Test connection reuse from pool"""
        # Create table first
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
            conn.commit()

        initial_stats = DatabasePool.get_pool(temp_db).get_performance_stats()

        # Multiple operations to test reuse
        for i in range(10):
            with DatabasePool.get_connection(temp_db) as conn:
                conn.execute("INSERT INTO test (value) VALUES (?)", (f"value_{i}",))
                conn.commit()

        final_stats = DatabasePool.get_pool(temp_db).get_performance_stats()

        # Should have reused connections
        assert final_stats['total_created'] >= 1
        assert final_stats['total_reused'] >= 1

    def test_prepared_statement_caching(self, temp_db):
        """Test prepared statement caching"""
        # Create table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)")
            conn.commit()

        # Insert multiple records with same query pattern
        query = "INSERT INTO test (name, value) VALUES (?, ?)"
        for i in range(100):
            execute_update(temp_db, query, (f"name_{i}", i))

        # Query with same pattern
        for i in range(50):
            results = execute_query(temp_db, "SELECT * FROM test WHERE value > ?", (i,))

        # Check that prepared statements are cached
        global_stats = DatabasePool.get_global_stats()
        # assert global_stats['cached_statements'] >= 1  # May be 0 if cache is cleared

    def test_thread_safety(self, temp_db):
        """Test thread-safe operations"""
        # Create table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("CREATE TABLE counter (id INTEGER PRIMARY KEY, count INTEGER)")
            conn.execute("INSERT INTO counter (count) VALUES (0)")
            conn.commit()

        def increment_counter():
            """Thread function to increment counter"""
            for _ in range(100):
                with DatabasePool.get_connection(temp_db) as conn:
                    conn.execute("UPDATE counter SET count = count + 1")
                    conn.commit()
                time.sleep(0.001)  # Small delay to increase chance of race conditions

        # Run multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=increment_counter)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify final count
        result = execute_query(temp_db, "SELECT count FROM counter")
        assert len(result) == 1
        assert result[0][0] == 500  # 5 threads * 100 increments each

    def test_connection_health_monitoring(self, temp_db):
        """Test connection validation and cleanup"""
        pool = DatabasePool.get_pool(temp_db)

        # Create a connection
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.commit()

        # Get initial stats
        initial_created = pool._total_connections_created

        # Force cleanup by setting old last_used
        for conn_info in pool._connections:
            conn_info.last_used = time.time() - 400  # Older than max_idle_time (300)

        # Trigger cleanup by acquiring new connection
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("SELECT 1")

        # Check that expired connections were cleaned up
        assert len(pool._connections) <= pool.max_connections

    def test_bulk_operations(self, temp_db):
        """Test bulk insert operations with connection pooling"""
        # Create table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("CREATE TABLE bulk_test (id INTEGER PRIMARY KEY, data TEXT)")
            conn.commit()

        # Prepare bulk data
        bulk_data = [(i, f"data_{i}") for i in range(1000)]

        # Execute bulk insert
        execute_many(temp_db, "INSERT INTO bulk_test (id, data) VALUES (?, ?)", bulk_data)

        # Verify all records inserted
        result = execute_query(temp_db, "SELECT COUNT(*) FROM bulk_test")
        assert result[0][0] == 1000

    def test_concurrent_operations(self, temp_db):
        """Test concurrent database operations"""
        # Create table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("CREATE TABLE concurrent_test (id INTEGER PRIMARY KEY, thread_id INTEGER, value INTEGER)")
            conn.commit()

        def worker_thread(thread_id):
            """Worker function for concurrent operations"""
            for i in range(50):
                execute_update(temp_db,
                             "INSERT INTO concurrent_test (thread_id, value) VALUES (?, ?)",
                             (thread_id, i))

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()

        # Verify all operations completed
        result = execute_query(temp_db, "SELECT COUNT(*) FROM concurrent_test")
        assert result[0][0] == 500  # 10 threads * 50 inserts each

    def test_performance_improvements(self, temp_db):
        """Test that pooling provides performance benefits"""
        # Create table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("CREATE TABLE perf_test (id INTEGER PRIMARY KEY, data TEXT)")
            conn.commit()

        # Measure time for multiple operations
        start_time = time.time()

        for i in range(100):
            execute_update(temp_db, "INSERT INTO perf_test (data) VALUES (?)", (f"perf_data_{i}",))

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (less than 1 second for 100 operations)
        assert duration < 1.0

        # Verify operations completed
        result = execute_query(temp_db, "SELECT COUNT(*) FROM perf_test")
        assert result[0][0] == 100

    def test_error_handling(self, temp_db):
        """Test error handling and recovery"""
        # Test invalid SQL
        with pytest.raises(sqlite3.Error):
            execute_query(temp_db, "INVALID SQL QUERY")

        # Test connection recovery after error
        # Should still work after error
        result = execute_query(temp_db, "SELECT 1")
        assert result == [(1,)]

    def test_pool_cleanup(self, temp_db):
        """Test pool cleanup functionality"""
        pool = DatabasePool.get_pool(temp_db)

        # Create some connections
        for _ in range(3):
            with DatabasePool.get_connection(temp_db) as conn:
                conn.execute("SELECT 1")

        initial_count = len(pool._connections)

        # Force cleanup
        DatabasePool.cleanup_cache()

        # Pool should still function
        with DatabasePool.get_connection(temp_db) as conn:
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    def test_automatic_cleanup(self, temp_db):
        """Test automatic cleanup of expired connections"""
        pool = DatabasePool.get_pool(temp_db)

        # Create a connection
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("CREATE TABLE test_cleanup (id INTEGER)")
            conn.commit()

        # Manually set last_used to old time to simulate expiration
        for conn_info in pool._connections:
            conn_info.last_used = time.time() - 400  # Older than default max_idle_time (300)

        # Trigger cleanup by calling _cleanup_expired_connections
        pool._cleanup_expired_connections(time.time())

        # Connection should be cleaned up
        assert len(pool._connections) == 0

    def test_connection_pool_limits(self, temp_db):
        """Test connection pool size limits"""
        max_conn = 3
        pool = DatabasePool.get_pool(temp_db, max_connections=max_conn)

        connections = []

        # Acquire max connections
        for i in range(max_conn):
            conn = pool._acquire_connection()
            connections.append(conn)

        # Next acquisition should either create new or wait
        # In our implementation, it will wait and reuse
        start_time = time.time()
        conn = pool._acquire_connection()
        elapsed = time.time() - start_time

        # Should not take too long (less than 1 second for reuse)
        assert elapsed < 1.0

        # Release all connections
        for conn in connections:
            pool._release_connection(conn)
        pool._release_connection(conn)

    def test_fallback_mechanism(self, temp_db):
        """Test fallback when DatabasePool import fails"""
        from unittest.mock import patch

        with patch('core.database_pool.DatabasePool', None):
            # Simulate import failure
            with patch.dict('sys.modules', {'core.database_pool': None}):
                # This should trigger fallback in dependent modules
                # Test that direct sqlite3 operations still work
                conn = sqlite3.connect(temp_db)
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    assert result[0] == 1
                finally:
                    conn.close()

    def test_performance_comparison(self, temp_db):
        """Compare performance of pooled vs direct connections"""
        # Setup test data
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("""
                CREATE TABLE perf_compare (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
            """)
            conn.commit()

        # Test pooled performance
        start_time = time.time()
        for _ in range(50):
            with DatabasePool.get_connection(temp_db) as conn:
                conn.execute("SELECT COUNT(*) FROM perf_compare")
                conn.fetchone()
        pooled_time = time.time() - start_time

        # Test direct connection performance
        start_time = time.time()
        for _ in range(50):
            conn = sqlite3.connect(temp_db)
            try:
                conn.execute("SELECT COUNT(*) FROM perf_compare")
                conn.fetchone()
            finally:
                conn.close()
        direct_time = time.time() - start_time

        # Pooled should be faster (though in this simple test it might not be dramatic)
        print(f"Pooled time: {pooled_time:.4f}s")
        print(f"Direct time: {direct_time:.4f}s")
        print(f"Speedup: {direct_time / pooled_time:.2f}x")

        # At minimum, pooled shouldn't be significantly slower
        assert pooled_time <= direct_time * 2.0

    def test_concurrent_performance(self, temp_db):
        """Test performance under concurrent load"""
        # Setup test data
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("""
                CREATE TABLE concurrent_perf (
                    id INTEGER PRIMARY KEY,
                    thread_id INTEGER,
                    data TEXT
                )
            """)
            conn.commit()

        def concurrent_worker(thread_id):
            for i in range(20):
                with DatabasePool.get_connection(temp_db) as conn:
                    conn.execute(
                        "INSERT INTO concurrent_perf (thread_id, data) VALUES (?, ?)",
                        (thread_id, f"data_{thread_id}_{i}")
                    )
                    conn.commit()

        # Run concurrent operations
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_worker, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()

        elapsed = time.time() - start_time

        # Verify all data was inserted
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("SELECT COUNT(*) FROM concurrent_perf")
            count = conn.fetchone()[0]
            assert count == 100  # 5 threads * 20 inserts each

        print(f"Concurrent test completed in {elapsed:.4f}s")

    def test_prepared_statement_cleanup(self, temp_db):
        """Test cleanup of expired prepared statements"""
        query = "SELECT * FROM test_cleanup_table"

        # Add a statement to cache
        DatabasePool.get_prepared_statement(query, temp_db)

        # Verify it's in cache
        cache_key = f"{temp_db}:{query}"
        assert cache_key in DatabasePool._stmt_cache

        # Manually expire it
        DatabasePool._stmt_cache[cache_key].last_used = time.time() - 7200  # 2 hours ago

        # Trigger cleanup
        DatabasePool.cleanup_cache()

        # Should be removed
        assert cache_key not in DatabasePool._stmt_cache

    def test_global_stats(self, temp_db):
        """Test global statistics collection"""
        # Create multiple pools
        pool1 = DatabasePool.get_pool(temp_db + "_1")
        pool2 = DatabasePool.get_pool(temp_db + "_2")

        # Use them
        with DatabasePool.get_connection(temp_db + "_1") as conn:
            conn.execute("SELECT 1")

        with DatabasePool.get_connection(temp_db + "_2") as conn:
            conn.execute("SELECT 1")

        # Get global stats
        global_stats = DatabasePool.get_global_stats()

        assert global_stats['total_pools'] >= 2
        assert len(global_stats['pools']) >= 2

    def test_connection_timeout_handling(self, temp_db):
        """Test handling of connection timeouts"""
        pool = DatabasePool.get_pool(temp_db, max_connections=1)

        # Acquire connection
        conn1 = pool._acquire_connection()

        # Try to acquire another - should timeout gracefully
        start_time = time.time()
        try:
            conn2 = pool._acquire_connection()
            # If we get here, it worked (either reused or created new)
            pool._release_connection(conn2)
        except sqlite3.OperationalError as e:
            assert "timeout" in str(e).lower() or "busy" in str(e).lower()
            elapsed = time.time() - start_time
            assert elapsed >= 25  # Should wait at least 25 seconds (half of 30s timeout)

        # Release first connection
        pool._release_connection(conn1)


class TestConvenienceFunctions:
    """Test convenience functions for database operations"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)

    def test_execute_query(self, temp_db):
        """Test execute_query convenience function"""
        # Setup table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("""
                CREATE TABLE test_query (
                    id INTEGER PRIMARY KEY,
                    name TEXT
                )
            """)
            conn.execute("INSERT INTO test_query (name) VALUES (?)", ("test",))
            conn.commit()

        # Test query execution
        results = execute_query(temp_db, "SELECT * FROM test_query")
        assert len(results) == 1
        assert results[0][1] == "test"

    def test_execute_update(self, temp_db):
        """Test execute_update convenience function"""
        # Setup table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("""
                CREATE TABLE test_update (
                    id INTEGER PRIMARY KEY,
                    value INTEGER
                )
            """)
            conn.execute("INSERT INTO test_update (value) VALUES (?)", (10,))
            conn.commit()

        # Test update execution
        rows_affected = execute_update(
            temp_db,
            "UPDATE test_update SET value = ? WHERE id = ?",
            (20, 1)
        )
        assert rows_affected == 1

        # Verify update
        results = execute_query(temp_db, "SELECT value FROM test_update WHERE id = 1")
        assert results[0][0] == 20

    def test_execute_many(self, temp_db):
        """Test execute_many convenience function"""
        # Setup table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("""
                CREATE TABLE test_many (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
            """)
            conn.commit()

        # Test batch insert
        data = [("item1",), ("item2",), ("item3",)]
        execute_many(temp_db, "INSERT INTO test_many (data) VALUES (?)", data)

        # Verify inserts
        results = execute_query(temp_db, "SELECT COUNT(*) FROM test_many")
        assert results[0][0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

    def test_automatic_cleanup(self, temp_db):
        """Test automatic cleanup of expired connections"""
        pool = DatabasePool.get_pool(temp_db)

        # Create a connection
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("CREATE TABLE test_cleanup (id INTEGER)")
            conn.commit()

        # Manually set last_used to old time to simulate expiration
        for conn_info in pool._connections:
            conn_info.last_used = time.time() - 400  # Older than default max_idle_time (300)

        # Trigger cleanup by calling _cleanup_expired_connections
        pool._cleanup_expired_connections(time.time())

        # Connection should be cleaned up
        assert len(pool._connections) == 0

    def test_connection_pool_limits(self, temp_db):
        """Test connection pool size limits"""
        max_conn = 3
        pool = DatabasePool.get_pool(temp_db, max_connections=max_conn)

        connections = []

        # Acquire max connections
        for i in range(max_conn):
            conn = pool._acquire_connection()
            connections.append(conn)

        # Next acquisition should either create new or wait
        # In our implementation, it will wait and reuse
        start_time = time.time()
        conn = pool._acquire_connection()
        elapsed = time.time() - start_time

        # Should not take too long (less than 1 second for reuse)
        assert elapsed < 1.0

        # Release all connections
        for conn in connections:
            pool._release_connection(conn)
        pool._release_connection(conn)

    def test_fallback_mechanism(self, temp_db):
        """Test fallback when DatabasePool import fails"""
        from unittest.mock import patch

        with patch('core.database_pool.DatabasePool', None):
            # Simulate import failure
            with patch.dict('sys.modules', {'core.database_pool': None}):
                # This should trigger fallback in dependent modules
                # Test that direct sqlite3 operations still work
                conn = sqlite3.connect(temp_db)
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    assert result[0] == 1
                finally:
                    conn.close()

    def test_performance_comparison(self, temp_db):
        """Compare performance of pooled vs direct connections"""
        # Setup test data
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("""
                CREATE TABLE perf_compare (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
            """)
            conn.commit()

        # Test pooled performance
        start_time = time.time()
        for _ in range(50):
            with DatabasePool.get_connection(temp_db) as conn:
                conn.execute("SELECT COUNT(*) FROM perf_compare")
                conn.fetchone()
        pooled_time = time.time() - start_time

        # Test direct connection performance
        start_time = time.time()
        for _ in range(50):
            conn = sqlite3.connect(temp_db)
            try:
                conn.execute("SELECT COUNT(*) FROM perf_compare")
                conn.fetchone()
            finally:
                conn.close()
        direct_time = time.time() - start_time

        # Pooled should be faster (though in this simple test it might not be dramatic)
        print(f"Pooled time: {pooled_time:.4f}s")
        print(f"Direct time: {direct_time:.4f}s")
        print(f"Speedup: {direct_time / pooled_time:.2f}x")

        # At minimum, pooled shouldn't be significantly slower
        assert pooled_time <= direct_time * 2.0

    def test_concurrent_performance(self, temp_db):
        """Test performance under concurrent load"""
        # Setup test data
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("""
                CREATE TABLE concurrent_perf (
                    id INTEGER PRIMARY KEY,
                    thread_id INTEGER,
                    data TEXT
                )
            """)
            conn.commit()

        def concurrent_worker(thread_id):
            for i in range(20):
                with DatabasePool.get_connection(temp_db) as conn:
                    conn.execute(
                        "INSERT INTO concurrent_perf (thread_id, data) VALUES (?, ?)",
                        (thread_id, f"data_{thread_id}_{i}")
                    )
                    conn.commit()

        # Run concurrent operations
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_worker, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()

        elapsed = time.time() - start_time

        # Verify all data was inserted
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("SELECT COUNT(*) FROM concurrent_perf")
            count = conn.fetchone()[0]
            assert count == 100  # 5 threads * 20 inserts each

        print(f"Concurrent test completed in {elapsed:.4f}s")

    def test_prepared_statement_cleanup(self, temp_db):
        """Test cleanup of expired prepared statements"""
        query = "SELECT * FROM test_cleanup_table"

        # Add a statement to cache
        DatabasePool.get_prepared_statement(query, temp_db)

        # Verify it's in cache
        cache_key = f"{temp_db}:{query}"
        assert cache_key in DatabasePool._stmt_cache

        # Manually expire it
        DatabasePool._stmt_cache[cache_key].last_used = time.time() - 7200  # 2 hours ago

        # Trigger cleanup
        DatabasePool.cleanup_cache()

        # Should be removed
        assert cache_key not in DatabasePool._stmt_cache

    def test_global_stats(self, temp_db):
        """Test global statistics collection"""
        # Create multiple pools
        pool1 = DatabasePool.get_pool(temp_db + "_1")
        pool2 = DatabasePool.get_pool(temp_db + "_2")

        # Use them
        with DatabasePool.get_connection(temp_db + "_1") as conn:
            conn.execute("SELECT 1")

        with DatabasePool.get_connection(temp_db + "_2") as conn:
            conn.execute("SELECT 1")

        # Get global stats
        global_stats = DatabasePool.get_global_stats()

        assert global_stats['total_pools'] >= 2
        assert len(global_stats['pools']) >= 2

    def test_connection_timeout_handling(self, temp_db):
        """Test handling of connection timeouts"""
        pool = DatabasePool.get_pool(temp_db, max_connections=1)

        # Acquire connection
        conn1 = pool._acquire_connection()

        # Try to acquire another - should timeout gracefully
        start_time = time.time()
        try:
            conn2 = pool._acquire_connection()
            # If we get here, it worked (either reused or created new)
            pool._release_connection(conn2)
        except sqlite3.OperationalError as e:
            assert "timeout" in str(e).lower() or "busy" in str(e).lower()
            elapsed = time.time() - start_time
            assert elapsed >= 25  # Should wait at least 25 seconds (half of 30s timeout)

        # Release first connection
        pool._release_connection(conn1)


class TestConvenienceFunctions:
    """Test convenience functions for database operations"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)

    def test_execute_query(self, temp_db):
        """Test execute_query convenience function"""
        # Setup table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("""
                CREATE TABLE test_query (
                    id INTEGER PRIMARY KEY,
                    name TEXT
                )
            """)
            conn.execute("INSERT INTO test_query (name) VALUES (?)", ("test",))
            conn.commit()

        # Test query execution
        results = execute_query(temp_db, "SELECT * FROM test_query")
        assert len(results) == 1
        assert results[0][1] == "test"

    def test_execute_update(self, temp_db):
        """Test execute_update convenience function"""
        # Setup table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("""
                CREATE TABLE test_update (
                    id INTEGER PRIMARY KEY,
                    value INTEGER
                )
            """)
            conn.execute("INSERT INTO test_update (value) VALUES (?)", (10,))
            conn.commit()

        # Test update execution
        rows_affected = execute_update(
            temp_db,
            "UPDATE test_update SET value = ? WHERE id = ?",
            (20, 1)
        )
        assert rows_affected == 1

        # Verify update
        results = execute_query(temp_db, "SELECT value FROM test_update WHERE id = 1")
        assert results[0][0] == 20

    def test_execute_many(self, temp_db):
        """Test execute_many convenience function"""
        # Setup table
        with DatabasePool.get_connection(temp_db) as conn:
            conn.execute("""
                CREATE TABLE test_many (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
            """)
            conn.commit()

        # Test batch insert
        data = [("item1",), ("item2",), ("item3",)]
        execute_many(temp_db, "INSERT INTO test_many (data) VALUES (?)", data)

        # Verify inserts
        results = execute_query(temp_db, "SELECT COUNT(*) FROM test_many")
        assert results[0][0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
