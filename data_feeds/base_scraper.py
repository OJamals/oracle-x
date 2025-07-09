from abc import ABC, abstractmethod
from typing import Any

class BaseScraper(ABC):
    """
    Abstract base class for all data feed scrapers.
    """
    @abstractmethod
    def fetch(self) -> Any:
        """Fetch data from the source."""
        pass
