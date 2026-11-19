"""Tests for the Intern Manager SQLite functions."""

import sqlite3

import database
import pytest


@pytest.fixture
def temporary_database(tmp_path, monkeypatch):
    """Initialize an isolated database under a temporary data directory."""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(database, "DATA_DIR", data_dir)
    monkeypatch.setattr(database, "DATABASE_PATH", data_dir / "interns.db")
    database.init_db()
    return data_dir / "interns.db"


def _intern_data(**overrides):
    data = {
        "full_name": "Ada Lovelace",
        "phone": "5551234567",
        "email": "ada@example.test",
        "college": "Analytical University",
        "department": "Mathematics",
        "skills": "Python, SQL",
        "status": "Active",
    }
    data.update(overrides)
    return data


def test_add_intern_creates_data_directory_and_returns_record(temporary_database):
    intern_id = database.add_intern(_intern_data())

    intern = database.get_intern(intern_id)
    assert temporary_database.exists()
    assert intern is not None
    assert intern["full_name"] == "Ada Lovelace"
    assert intern["email"] == "ada@example.test"
    assert intern["skills"] == "Python, SQL"
    assert intern["created_at"]


def test_search_interns_matches_name_phone_and_email_partially(temporary_database):
    intern_id = database.add_intern(_intern_data())

    assert [intern["id"] for intern in database.search_interns("ADA")] == [intern_id]
    assert [intern["id"] for intern in database.search_interns("1234")] == [intern_id]
    assert [intern["id"] for intern in database.search_interns("EXAMPLE")] == [intern_id]
    assert database.search_interns("not-found") == []


def test_update_intern_changes_fields_and_reports_missing_record(temporary_database):
    intern_id = database.add_intern(_intern_data())

    assert database.update_intern(
        intern_id,
        {"department": "Computer Science", "status": "Completed"},
    )
    assert database.get_intern(intern_id)["department"] == "Computer Science"
    assert database.get_intern(intern_id)["status"] == "Completed"
    assert not database.update_intern(intern_id + 1, {"notes": "missing"})


def test_delete_intern_cascades_to_attached_files(temporary_database):
    intern_id = database.add_intern(_intern_data())
    file_id = database.add_file(
        intern_id,
        "resume.pdf",
        "uploads/resume.pdf",
        "Resume",
    )

    assert database.get_files(intern_id)[0]["id"] == file_id
    assert database.delete_intern(intern_id)
    assert database.get_intern(intern_id) is None
    assert database.get_files(intern_id) == []
    assert not database.delete_intern(intern_id)


def test_delete_file_removes_only_the_selected_file(temporary_database):
    intern_id = database.add_intern(_intern_data())
    file_id = database.add_file(
        intern_id,
        "resume.pdf",
        "uploads/resume.pdf",
        "Resume",
    )

    assert database.delete_file(file_id)
    assert database.get_files(intern_id) == []
    assert not database.delete_file(file_id)


def test_init_db_adds_skills_column_to_existing_database(tmp_path, monkeypatch):
    data_dir = tmp_path / "legacy-data"
    data_dir.mkdir()
    database_path = data_dir / "interns.db"
    monkeypatch.setattr(database, "DATA_DIR", data_dir)
    monkeypatch.setattr(database, "DATABASE_PATH", database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE interns (id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT)"
        )

    database.init_db()

    with database.get_connection() as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(interns)")
        }
    assert "skills" in columns
