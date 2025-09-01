"""
Async I/O Utilities for Oracle-X Performance Optimization

Provides async versions of common I/O operations for improved throughput:
- AsyncFileManager: Async file operations using aiofiles
- AsyncDatabaseManager: Async database operations using aiosqlite
- AsyncHTTPClient: Async HTTP operations using aiohttp
- AsyncIOManager: Unified interface for all async I/O operations

Usage:
    from core.async_io_utils import AsyncIOManager

    async with AsyncIOManager() as io_manager:
        # Async file operations
        data = await io_manager.file.read_json('data.json')

        # Async database operations
        results = await io_manager.db.execute_query('SELECT * FROM table')

        # Async HTTP operations
        response = await io_manager.http.get('https://api.example.com/data')
"""

import asyncio
import json
import os
import logging
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
from pathlib import Path

# Async I/O libraries with fallbacks
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

# Fallback imports for sync operations
import sqlite3
import requests

logger = logging.getLogger(__name__)


class AsyncFileManager:
    """Async file operations manager using aiofiles"""

    def __init__(self):
        if not AIOFILES_AVAILABLE:
            logger.warning("aiofiles not available, falling back to sync file operations")
            self.async_available = False
        else:
            self.async_available = True

    async def read_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Read JSON file asynchronously"""
        if not self.async_available or aiofiles is None:
            return self._read_json_sync(file_path)

        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to read JSON file {file_path}: {e}")
            return None

    async def write_json(self, file_path: Union[str, Path], data: Dict[str, Any], indent: int = 2) -> bool:
        """Write JSON file asynchronously"""
        if not self.async_available or aiofiles is None:
            return self._write_json_sync(file_path, data, indent)

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                content = json.dumps(data, indent=indent, default=str)
                await f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write JSON file {file_path}: {e}")
            return False

    async def read_text(self, file_path: Union[str, Path]) -> Optional[str]:
        """Read text file asynchronously"""
        if not self.async_available or aiofiles is None:
            return self._read_text_sync(file_path)

        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return None

    async def write_text(self, file_path: Union[str, Path], content: str) -> bool:
        """Write text file asynchronously"""
        if not self.async_available or aiofiles is None:
            return self._write_text_sync(file_path, content)

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write text file {file_path}: {e}")
            return False

    # Sync fallback methods
    def _read_json_sync(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Sync fallback for reading JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read JSON file {file_path}: {e}")
            return None

    def _write_json_sync(self, file_path: Union[str, Path], data: Dict[str, Any], indent: int = 2) -> bool:
        """Sync fallback for writing JSON"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to write JSON file {file_path}: {e}")
            return False

    def _read_text_sync(self, file_path: Union[str, Path]) -> Optional[str]:
        """Sync fallback for reading text"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return None

    def _write_text_sync(self, file_path: Union[str, Path], content: str) -> bool:
        """Sync fallback for writing text"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write text file {file_path}: {e}")
            return False


class AsyncDatabaseManager:
    """Async database operations manager using aiosqlite"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        if not AIOSQLITE_AVAILABLE:
            logger.warning("aiosqlite not available, falling back to sync database operations")
            self.async_available = False
        else:
            self.async_available = True

    @asynccontextmanager
    async def connection(self):
        """Async context manager for database connections"""
        if not self.async_available or aiosqlite is None:
            # Sync fallback
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()
            return

        conn = await aiosqlite.connect(self.db_path)
        try:
            yield conn
        finally:
            await conn.close()

    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results as list of dicts"""
        if not self.async_available or aiosqlite is None:
            return self._execute_query_sync(query, params)

        async with self.connection() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description or []]

            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))

            await cursor.close()
            return results

    async def execute_write(self, query: str, params: tuple = ()) -> bool:
        """Execute INSERT/UPDATE/DELETE query"""
        if not self.async_available or aiosqlite is None:
            return self._execute_write_sync(query, params)

        async with self.connection() as conn:
            try:
                await conn.execute(query, params)
                await conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to execute write query: {e}")
                await conn.rollback()
                return False

    async def execute_many(self, query: str, params_list: List[tuple]) -> bool:
        """Execute multiple INSERT/UPDATE/DELETE queries"""
        if not self.async_available or aiosqlite is None:
            return self._execute_many_sync(query, params_list)

        async with self.connection() as conn:
            try:
                await conn.executemany(query, params_list)
                await conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to execute many query: {e}")
                await conn.rollback()
                return False

    # Sync fallback methods
    def _execute_query_sync(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Sync fallback for SELECT queries"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description or []]

            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))

            return results
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []
        finally:
            conn.close()

    def _execute_write_sync(self, query: str, params: tuple = ()) -> bool:
        """Sync fallback for write queries"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(query, params)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to execute write query: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def _execute_many_sync(self, query: str, params_list: List[tuple]) -> bool:
        """Sync fallback for multiple write queries"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(query, params_list)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to execute many query: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()


class AsyncHTTPClient:
    """Async HTTP operations manager using aiohttp"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, falling back to sync HTTP operations")
            self.async_available = False
        else:
            self.async_available = True
            self._session = None

    async def __aenter__(self):
        if self.async_available and aiohttp is not None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def get(self, url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Async GET request"""
        if not self.async_available or aiohttp is None:
            return self._get_sync(url, headers, params)

        if not self._session:
            raise RuntimeError("HTTP client not properly initialized. Use 'async with' context manager.")

        try:
            async with self._session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"HTTP GET failed: {response.status} - {url}")
                    return None
        except Exception as e:
            logger.error(f"HTTP GET error for {url}: {e}")
            return None

    async def post(self, url: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Async POST request"""
        if not self.async_available or aiohttp is None:
            return self._post_sync(url, data, headers)

        if not self._session:
            raise RuntimeError("HTTP client not properly initialized. Use 'async with' context manager.")

        try:
            async with self._session.post(url, json=data, headers=headers) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    logger.error(f"HTTP POST failed: {response.status} - {url}")
                    return None
        except Exception as e:
            logger.error(f"HTTP POST error for {url}: {e}")
            return None

    # Sync fallback methods
    def _get_sync(self, url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Sync fallback for GET requests"""
        try:
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"HTTP GET failed: {response.status_code} - {url}")
                return None
        except Exception as e:
            logger.error(f"HTTP GET error for {url}: {e}")
            return None

    def _post_sync(self, url: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Sync fallback for POST requests"""
        try:
            response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
            if response.status_code in [200, 201]:
                return response.json()
            else:
                logger.error(f"HTTP POST failed: {response.status_code} - {url}")
                return None
        except Exception as e:
            logger.error(f"HTTP POST error for {url}: {e}")
            return None


class AsyncIOManager:
    """Unified async I/O manager providing access to all async operations"""

    def __init__(self):
        self.file = AsyncFileManager()
        self.db = None  # Will be set when database path is provided
        self.http = AsyncHTTPClient()

    def set_database(self, db_path: str):
        """Set database path for async database operations"""
        self.db = AsyncDatabaseManager(db_path)

    @asynccontextmanager
    async def http_session(self):
        """Context manager for HTTP operations"""
        async with self.http:
            yield self.http

    async def concurrent_http_requests(self, requests_list: List[Dict[str, Any]]) -> List[Any]:
        """Execute multiple HTTP requests concurrently"""
        async with self.http:
            tasks = []
            for req in requests_list:
                if req['method'].upper() == 'GET':
                    task = self.http.get(req['url'], req.get('headers'), req.get('params'))
                elif req['method'].upper() == 'POST':
                    task = self.http.post(req['url'], req.get('data'), req.get('headers'))
                else:
                    task = asyncio.create_task(asyncio.sleep(0))  # No-op for unsupported methods
                tasks.append(task)

            return await asyncio.gather(*tasks, return_exceptions=True)

    async def concurrent_file_operations(self, operations: List[Dict[str, Any]]) -> List[bool]:
        """Execute multiple file operations concurrently"""
        tasks = []
        for op in operations:
            if op['type'] == 'read_json':
                task = self.file.read_json(op['path'])
            elif op['type'] == 'write_json':
                task = self.file.write_json(op['path'], op['data'], op.get('indent', 2))
            elif op['type'] == 'read_text':
                task = self.file.read_text(op['path'])
            elif op['type'] == 'write_text':
                task = self.file.write_text(op['path'], op['content'])
            else:
                task = asyncio.create_task(asyncio.sleep(0))  # No-op for unsupported operations
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert results to boolean success indicators
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(False)
            elif isinstance(result, dict) or isinstance(result, str) or result is None:
                processed_results.append(True)
            else:
                processed_results.append(bool(result))

        return processed_results


# Global instance for easy access
async_io_manager = AsyncIOManager()


async def get_async_io_manager(db_path: Optional[str] = None) -> AsyncIOManager:
    """Get configured async I/O manager instance"""
    if db_path:
        async_io_manager.set_database(db_path)
    return async_io_manager
