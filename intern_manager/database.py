"""SQLite database functions for Intern Manager."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent / "data"
DATABASE_PATH = DATA_DIR / "interns.db"

_INTERN_FIELDS = (
    "full_name",
    "phone",
    "email",
    "college",
    "department",
    "skills",
    "start_date",
    "end_date",
    "status",
    "notes",
    "photo_path",
)
_FILE_TYPES = ("Resume", "Certificate", "ID", "Other")


def ensure_data_directory() -> None:
    """Create the data directory if it does not exist."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not create database data directory")
        raise


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with foreign-key enforcement enabled."""
    ensure_data_directory()
    try:
        connection = sqlite3.connect(DATABASE_PATH)
    except sqlite3.Error:
        logger.exception("Could not open SQLite database")
        raise
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
    except sqlite3.Error:
        connection.close()
        logger.exception("Could not configure SQLite connection")
        raise
    return connection


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """Yield a connection and ensure it is closed after the transaction."""
    connection = get_connection()
    try:
        with connection:
            yield connection
    except sqlite3.Error:
        logger.exception("SQLite transaction failed")
        raise
    finally:
        connection.close()


def init_db() -> None:
    """Create the interns and intern_files tables if they do not exist."""
    ensure_data_directory()
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT,
                phone TEXT,
                email TEXT UNIQUE,
                college TEXT,
                department TEXT,
                skills TEXT,
                start_date TEXT,
                end_date TEXT,
                status TEXT CHECK (status IN ('Active', 'Completed')),
                notes TEXT,
                photo_path TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        intern_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(interns)")
        }
        if "skills" not in intern_columns:
            connection.execute("ALTER TABLE interns ADD COLUMN skills TEXT")
            logger.info("Added skills column to existing interns table")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS intern_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intern_id INTEGER NOT NULL,
                file_name TEXT,
                file_path TEXT,
                file_type TEXT CHECK (
                    file_type IN ('Resume', 'Certificate', 'ID', 'Other')
                ),
                uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (intern_id) REFERENCES interns (id) ON DELETE CASCADE
            )
            """
        )
    logger.info("SQLite database schema initialized")


def _validate_fields(data: dict[str, Any], allowed_fields: tuple[str, ...]) -> None:
    """Reject fields that are not part of the target table."""
    unknown_fields = set(data) - set(allowed_fields)
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unsupported field(s): {names}")


def add_intern(data: dict[str, Any]) -> int:
    """Insert an intern and return its database ID."""
    _validate_fields(data, _INTERN_FIELDS)
    if not data:
        raise ValueError("Intern data must contain at least one field.")

    fields = tuple(data)
    placeholders = ", ".join("?" for _ in fields)
    columns = ", ".join(fields)
    with _connection() as connection:
        cursor = connection.execute(
            f"INSERT INTO interns ({columns}) VALUES ({placeholders})",
            tuple(data[field] for field in fields),
        )
        intern_id = int(cursor.lastrowid)
    logger.info("Added intern record id=%s", intern_id)
    return intern_id


def update_intern(intern_id: int, data: dict[str, Any]) -> bool:
    """Update an intern and return whether a record was changed."""
    _validate_fields(data, _INTERN_FIELDS)
    if not data:
        raise ValueError("Intern data must contain at least one field.")

    assignments = ", ".join(f"{field} = ?" for field in data)
    values = tuple(data.values()) + (intern_id,)
    with _connection() as connection:
        cursor = connection.execute(
            f"UPDATE interns SET {assignments} WHERE id = ?",
            values,
        )
        updated = cursor.rowcount > 0
    if updated:
        logger.info("Updated intern record id=%s", intern_id)
    return updated


def delete_intern(intern_id: int) -> bool:
    """Delete an intern and return whether a record was deleted."""
    with _connection() as connection:
        cursor = connection.execute("DELETE FROM interns WHERE id = ?", (intern_id,))
        deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Deleted intern record id=%s", intern_id)
    return deleted


def get_intern(intern_id: int) -> dict[str, Any] | None:
    """Retrieve an intern by ID, or return None if it does not exist."""
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM interns WHERE id = ?", (intern_id,)
        ).fetchone()
        return dict(row) if row is not None else None


def get_all_interns() -> list[dict[str, Any]]:
    """Retrieve all interns ordered by ID."""
    with _connection() as connection:
        rows = connection.execute("SELECT * FROM interns ORDER BY id").fetchall()
        return [dict(row) for row in rows]


def search_interns(query: str) -> list[dict[str, Any]]:
    """Find interns with a case-insensitive partial match on name, phone, or email."""
    pattern = f"%{query}%"
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM interns
            WHERE full_name LIKE ? COLLATE NOCASE
               OR phone LIKE ? COLLATE NOCASE
               OR email LIKE ? COLLATE NOCASE
            ORDER BY id
            """,
            (pattern, pattern, pattern),
        ).fetchall()
        results = [dict(row) for row in rows]
    logger.debug("Intern search returned %s record(s)", len(results))
    return results


def add_file(
    intern_id: int,
    file_name: str,
    file_path: str,
    file_type: str,
) -> int:
    """Add an intern file and return its database ID."""
    if file_type not in _FILE_TYPES:
        raise ValueError(f"file_type must be one of: {', '.join(_FILE_TYPES)}")

    with _connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO intern_files (intern_id, file_name, file_path, file_type)
            VALUES (?, ?, ?, ?)
            """,
            (intern_id, file_name, file_path, file_type),
        )
        file_id = int(cursor.lastrowid)
    logger.info("Added file record id=%s for intern_id=%s", file_id, intern_id)
    return file_id


def get_files(intern_id: int) -> list[dict[str, Any]]:
    """Retrieve files belonging to an intern, ordered by ID."""
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM intern_files WHERE intern_id = ? ORDER BY id",
            (intern_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def delete_file(file_id: int) -> bool:
    """Delete an intern file and return whether a record was deleted."""
    with _connection() as connection:
        cursor = connection.execute("DELETE FROM intern_files WHERE id = ?", (file_id,))
        deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Deleted file record id=%s", file_id)
    return deleted
