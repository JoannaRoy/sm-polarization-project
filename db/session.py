"""Database session: open the SQLite database and ensure the schema exists."""

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from config import DB_PATH
from db.models import Base

_engine_cache = {}


def _build_engine(db_path):
    if db_path == ":memory:":
        url = "sqlite:///:memory:"
    else:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
    engine = create_engine(url)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


def _get_engine(db_path):
    # In-memory SQLite databases are per-connection, so caching a shared engine
    # would let unrelated callers see each other's tables. Build a fresh engine
    # for every :memory: call instead.
    if db_path == ":memory:":
        return _build_engine(db_path)
    engine = _engine_cache.get(db_path)
    if engine is None:
        engine = _build_engine(db_path)
        _engine_cache[db_path] = engine
    return engine


def connect(db_path=DB_PATH):
    return Session(_get_engine(db_path))
