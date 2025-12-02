#!/usr/bin/env python3
"""
Root-level CLI Validation entrypoint.

This thin wrapper delegates to tests/cli_validate.py so tests that invoke:
  python cli_validate.py ...
from the repository root receive structured JSON output as expected.

Do not add any heavy logic here; keep behavior centralized in tests/cli_validate.py.
"""

from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    from tests.cli_validate import main as _main

    _main(argv)


if __name__ == "__main__":
    main()
