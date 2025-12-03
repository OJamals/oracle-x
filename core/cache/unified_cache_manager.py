"""
Unified Cache Manager for ORACLE-X
Multi-backend (memory, SQLite, Redis), config-driven, async, batch ops.
Preserves TTL, fallback, performance from existing implementations.
"""

import asyncio
import json
import logging
import time

from collections import OrderedDict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import sqlite3

try:
    import redis

    REDIS_AVAILABLE

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from core.config import config
from core.types_internal import DataSource

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in order of speed"""

    MEMORY

    MEMORY = 1
    SQLITE = 2
    REDIS = 3


@dataclass
class CacheEntry:
    """Cached item metadata"""

    key: str
