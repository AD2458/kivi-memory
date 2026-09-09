"""SQLite connection management. One file, WAL mode, foreign keys on."""
import sqlite3
from pathlib import Path
from contextlib import contextmanager

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "kivi.db"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH, reset: bool = False) -> None:
    """Create tables if absent. If reset=True, drop and recreate everything
    -- this backs the assignment's required 'reset the system' control."""
    path = Path(db_path)
    if reset and path.exists():
        path.unlink()
        for suffix in ("-wal", "-shm"):
            side = Path(str(path) + suffix)
            if side.exists():
                side.unlink()

    conn = get_connection(path)
    try:
        with open(SCHEMA_PATH, "r") as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


@contextmanager
def db_session(db_path: str | Path = DEFAULT_DB_PATH):
    """Context manager that commits on success, rolls back on exception."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
