#!/usr/bin/env python3
"""Deprecated alias for scripts/nightly_sgd_update.py (kept for older notes)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nightly_sgd_update import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
