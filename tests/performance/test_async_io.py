"""
Test Async I/O Operations
Tests async file operations, async database operations, async HTTP operations, and fallback mechanisms.
"""

import pytest
import asyncio
import tempfile
import os
import json
import sqlite3
from unittest.mock import Mock, patch, AsyncMock
from core.async_io_utils import (
    AsyncFileManager, AsyncDatabaseManager, AsyncHTTPClient,
    AsyncIOManager, get_async_io_manager
)


class TestAsyncFileManager:
    """Test suite for AsyncFileManager"""

    @pytest.fixture
    def file_manager(self):
        """Get AsyncFileManager instance"""
        return AsyncFileManager()

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create temporary file for testing"""
        temp_file = tmp_path / "test.json"
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        temp_file.write_text(json.dumps(test_data))
        return temp_file

    @pytest.mark.asyncio
    async def test_read_json_async(self, file_manager, temp_file):
        """Test async JSON file reading"""
        if not file_manager.async_available:
            pytest.skip("aiofiles not available")

        data = await file_manager.read_json(str(temp_file))
        assert data == {"key": "value", "numbers": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_write_json_async(self, file_manager, tmp_path):
        """Test async JSON file writing"""
        if not file_manager.async_available:
            pytest.skip("aiofiles not available")

        test_data = {"test": "data", "array": [4, 5, 6]}
        file_path = tmp_path / "write_test.json"

        success = await file_manager.write_json(str(file_path), test_data)
        assert success == True

        # Verify file was written
        with open(file_path, 'r') as f:
            written_data = json.load(f)
        assert written_data == test_data

    @pytest.mark.asyncio
    async def test_read_text_async(self, file_manager, tmp_path):
        """Test async text file reading"""
        if not file_manager.async_available:
            pytest.skip("aiofiles not available")

        test_content = "Hello, async world!"
        text_file = tmp_path / "test.txt"
        text_file.write_text(test_content)

        content = await file_manager.read_text(str(text_file))
        assert content == test_content

    @pytest.mark.asyncio
    async def test_write_text_async(self, file_manager, tmp_path):
        """Test async text file writing"""
        if not file_manager.async_available:
            pytest.skip("aiofiles not available")

        test_content = "Async file writing test"
        file_path = tmp_path / "write_text.txt"

        success = await file_manager.write_text(str(file_path), test_content)
        assert success == True

        # Verify file was written
        with open(file_path, 'r') as f:
            written_content = f.read()
        assert written_content == test_content

    def test_sync_fallback_json(self, file_manager, temp_file):
        """Test sync fallback for JSON operations"""
        # Force sync fallback
        file_manager.async_available = False

        # This should work with sync fallback
        import asyncio
        async def test_sync():
            data = await file_manager.read_json(str(temp_file))
            return data

        result = asyncio.run(test_sync())
        assert result == {"key": "value", "numbers": [1, 2, 3]}

    def test_sync_fallback_text(self, file_manager, tmp_path):
        """Test sync fallback for text operations"""
        file_manager.async_available = False

        test_content = "Sync fallback test"
        text_file = tmp_path / "sync_test.txt"
        text_file.write_text(test_content)

        async def test_sync():
            content = await file_manager.read_text(str(text_file))
            return content

        result = asyncio.run(test_sync())
        assert result == test_content


class TestAsyncDatabaseManager:
    """Test suite for AsyncDatabaseManager"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)

    @pytest.fixture
    def db_manager(self, temp_db):
        """Get AsyncDatabaseManager instance"""
        return AsyncDatabaseManager(temp_db)

    @pytest.mark.asyncio
    async def test_execute_query_async(self, db_manager):
        """Test async query execution"""
        if not db_manager.async_available:
            pytest.skip("aiosqlite not available")

        # Create test table
        async with db_manager.connection() as conn:
            await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
            await conn.execute("INSERT INTO test (name) VALUES ('Alice'), ('Bob')")
            await conn.commit()

        # Execute query
        results = await db_manager.execute_query("SELECT * FROM test ORDER BY id")

        expected = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]
        assert results == expected

    @pytest.mark.asyncio
    async def test_execute_write_async(self, db_manager):
        """Test async write operations"""
        if not db_manager.async_available:
            pytest.skip("aiosqlite not available")

        # Create test table and insert data
        success = await db_manager.execute_write(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)"
        )
        assert success == True

        success = await db_manager.execute_write(
            "INSERT INTO users (email) VALUES (?)", ("test@example.com",)
        )
        assert success == True

        # Verify data was inserted
        results = await db_manager.execute_query("SELECT * FROM users")
        assert len(results) == 1
        assert results[0]["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_execute_many_async(self, db_manager):
        """Test async bulk insert operations"""
        if not db_manager.async_available:
            pytest.skip("aiosqlite not available")

        # Create test table
        await db_manager.execute_write("CREATE TABLE bulk_test (id INTEGER PRIMARY KEY, value INTEGER)")

        # Bulk insert data
        data = [(i, i * 2) for i in range(100)]
        success = await db_manager.execute_many(
            "INSERT INTO bulk_test (id, value) VALUES (?, ?)", data
        )
        assert success == True

        # Verify all data was inserted
        results = await db_manager.execute_query("SELECT COUNT(*) as count FROM bulk_test")
        assert results[0]["count"] == 100

    def test_sync_fallback_query(self, db_manager):
        """Test sync fallback for queries"""
        db_manager.async_available = False

        async def test_sync():
            results = await db_manager.execute_query("SELECT 1 as test")
            return results

        result = asyncio.run(test_sync())
        assert result == [{"test": 1}]

    def test_sync_fallback_write(self, db_manager):
        """Test sync fallback for write operations"""
        db_manager.async_available = False

        async def test_sync():
            success = await db_manager.execute_write("CREATE TABLE sync_test (id INTEGER)")
            return success

        result = asyncio.run(test_sync())
        assert result == True


class TestAsyncHTTPClient:
    """Test suite for AsyncHTTPClient"""

    @pytest.fixture
    def http_client(self):
        """Get AsyncHTTPClient instance"""
        return AsyncHTTPClient(timeout=10)

    @pytest.mark.asyncio
    async def test_get_request_async(self, http_client):
        """Test async GET requests"""
        if not http_client.async_available:
            pytest.skip("aiohttp not available")

        # Mock the aiohttp response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"data": "test"})
            mock_get.return_value.__aenter__.return_value = mock_response

            async with http_client:
                result = await http_client.get("https://api.example.com/data")

            assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_post_request_async(self, http_client):
        """Test async POST requests"""
        if not http_client.async_available:
            pytest.skip("aiohttp not available")

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json = AsyncMock(return_value={"id": 1, "created": True})
            mock_post.return_value.__aenter__.return_value = mock_response

            async with http_client:
                result = await http_client.post("https://api.example.com/create", {"name": "test"})

            assert result == {"id": 1, "created": True}

    def test_sync_fallback_get(self, http_client):
        """Test sync fallback for GET requests"""
        http_client.async_available = False

        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"fallback": "data"}
            mock_get.return_value = mock_response

            async def test_sync():
                return await http_client.get("https://api.example.com/data")

            result = asyncio.run(test_sync())
            assert result == {"fallback": "data"}

    def test_sync_fallback_post(self, http_client):
        """Test sync fallback for POST requests"""
        http_client.async_available = False

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"posted": True}
            mock_post.return_value = mock_response

            async def test_sync():
                return await http_client.post("https://api.example.com/post", {"data": "test"})

            result = asyncio.run(test_sync())
            assert result == {"posted": True}


class TestAsyncIOManager:
    """Test suite for AsyncIOManager"""

    @pytest.fixture
    def io_manager(self):
        """Get AsyncIOManager instance"""
        return AsyncIOManager()

    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)

    def test_initialization(self, io_manager):
        """Test AsyncIOManager initialization"""
        assert io_manager.file is not None
        assert io_manager.http is not None
        assert io_manager.db is None  # Not set initially

    def test_set_database(self, io_manager, temp_db):
        """Test setting database path"""
        io_manager.set_database(temp_db)
        assert io_manager.db is not None
        assert io_manager.db.db_path == temp_db

    @pytest.mark.asyncio
    async def test_concurrent_http_requests(self, io_manager):
        """Test concurrent HTTP requests"""
        if not io_manager.http.async_available:
            pytest.skip("aiohttp not available")

        requests_list = [
            {"method": "GET", "url": "https://api1.example.com"},
            {"method": "GET", "url": "https://api2.example.com"},
            {"method": "POST", "url": "https://api3.example.com", "data": {"test": True}}
        ]

        with patch('aiohttp.ClientSession.get') as mock_get, \
             patch('aiohttp.ClientSession.post') as mock_post:

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"success": True})

            mock_get.return_value.__aenter__.return_value = mock_response
            mock_post.return_value.__aenter__.return_value = mock_response

            async with io_manager.http_session():
                results = await io_manager.concurrent_http_requests(requests_list)

            assert len(results) == 3
            assert all(result == {"success": True} for result in results if not isinstance(result, Exception))

    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self, io_manager, tmp_path):
        """Test concurrent file operations"""
        if not io_manager.file.async_available:
            pytest.skip("aiofiles not available")

        operations = [
            {"type": "write_json", "path": str(tmp_path / "file1.json"), "data": {"id": 1}},
            {"type": "write_json", "path": str(tmp_path / "file2.json"), "data": {"id": 2}},
            {"type": "write_text", "path": str(tmp_path / "file3.txt"), "content": "test content"}
        ]

        results = await io_manager.concurrent_file_operations(operations)

        assert len(results) == 3
        assert all(result == True for result in results)

        # Verify files were created
        assert (tmp_path / "file1.json").exists()
        assert (tmp_path / "file2.json").exists()
        assert (tmp_path / "file3.txt").exists()

    def test_get_async_io_manager(self):
        """Test get_async_io_manager function"""
        manager = asyncio.run(get_async_io_manager())
        assert isinstance(manager, AsyncIOManager)

    def test_get_async_io_manager_with_db(self, temp_db):
        """Test get_async_io_manager with database path"""
        manager = asyncio.run(get_async_io_manager(temp_db))
        assert isinstance(manager, AsyncIOManager)
        assert manager.db is not None
        assert manager.db.db_path == temp_db


class TestIntegration:
    """Integration tests for async I/O components"""

    @pytest.mark.asyncio
    async def test_full_async_workflow(self, tmp_path):
        """Test complete async workflow"""
        # Create test database
        db_path = str(tmp_path / "test.db")
        db_manager = AsyncDatabaseManager(db_path)

        if db_manager.async_available:
            # Create table
            await db_manager.execute_write("""
                CREATE TABLE workflow_test (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
            """)

            # Insert data
            await db_manager.execute_write(
                "INSERT INTO workflow_test (data) VALUES (?)",
                ("async workflow test",)
            )

            # Query data
            results = await db_manager.execute_query("SELECT * FROM workflow_test")
            assert len(results) == 1
            assert results[0]["data"] == "async workflow test"

    def test_fallback_availability(self):
        """Test that fallback mechanisms are always available"""
        # These should always work regardless of async library availability
        file_manager = AsyncFileManager()
        db_manager = AsyncDatabaseManager(":memory:")
        http_client = AsyncHTTPClient()

        # File operations should have sync fallbacks
        assert hasattr(file_manager, '_read_json_sync')
        assert hasattr(file_manager, '_write_json_sync')

        # Database operations should have sync fallbacks
        assert hasattr(db_manager, '_execute_query_sync')
        assert hasattr(db_manager, '_execute_write_sync')

        # HTTP operations should have sync fallbacks
        assert hasattr(http_client, '_get_sync')
        assert hasattr(http_client, '_post_sync')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])