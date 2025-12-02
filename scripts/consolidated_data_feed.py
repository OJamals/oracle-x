"""
Shim module to preserve legacy imports.

This file re-exports the public API from data_feeds.consolidated_data_feed to
eliminate import ambiguity with the package module and avoid sys.path hacks.

Consumers should prefer importing from data_feeds.consolidated_data_feed:
    from data_feeds.consolidated_data_feed import ConsolidatedDataFeed

This shim exists for backward compatibility only.
"""

from importlib import import_module as _import_module
import inspect as _inspect

# Import the real implementation module
_mod = _import_module("data_feeds.consolidated_data_feed")

# Re-export everything public from the real module
from data_feeds.consolidated_data_feed import *  # noqa: F401,F403

# If the target module defines __all__, mirror it. Otherwise, construct a safe public set.
if hasattr(_mod, "__all__"):
    __all__ = list(_mod.__all__)  # type: ignore[attr-defined]
else:
    __all__ = [
        name
        for name, obj in _mod.__dict__.items()
        if not name.startswith("_")
        and not (hasattr(obj, "__package__") and _inspect.ismodule(obj))
    ]
