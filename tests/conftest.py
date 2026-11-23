"""Shared fixtures: throwaway SQLite, simulated sensors, one ingest token.

Every test runs against a temporary database so `pytest` never touches the
deployment's fungi.db, and the sensors are simulated because CI has no DHT22.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def app_client():
    tmp = tempfile.TemporaryDirectory(prefix="mushy-tests-")
    os.environ["FUNGI_DB"] = str(Path(tmp.name) / "test.db")
    os.environ["FUNGI_SIMULATE"] = "1"
    os.environ["MUSHY_INGEST_TOKEN"] = "test-token"

    from app.server import create_app

    application = create_app()
    application.config.update(TESTING=True)
    client = application.test_client()
    yield client
    tmp.cleanup()
    for key in ("FUNGI_DB", "FUNGI_SIMULATE", "MUSHY_INGEST_TOKEN"):
        os.environ.pop(key, None)


@pytest.fixture()
def fresh_db(monkeypatch, tmp_path):
    monkeypatch.setenv("FUNGI_DB", str(tmp_path / "unit.db"))
    import app.db as db

    db.init_db()
    return db
