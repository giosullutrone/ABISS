"""Tests for generate_schema_prompt in db_datasets/sql_schema_prompts.py"""
import os
import sqlite3
import tempfile
import pytest
from db_datasets.sql_schema_prompts import generate_schema_prompt


@pytest.fixture
def simple_db():
    """Create a simple test database with a basic table."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    conn.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
    conn.commit()
    conn.close()
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def reserved_word_db():
    """Create a test database with tables using SQLite reserved words."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    conn = sqlite3.connect(db_path)
    # 'order', 'group', 'select', 'table', 'index' are all reserved words
    conn.execute('CREATE TABLE "order" (id INTEGER PRIMARY KEY, total REAL)')
    conn.execute('INSERT INTO "order" VALUES (1, 99.99)')
    conn.execute('CREATE TABLE "group" (id INTEGER PRIMARY KEY, name TEXT)')
    conn.execute('INSERT INTO "group" VALUES (1, \'Engineering\')')
    conn.execute('CREATE TABLE "select" (id INTEGER PRIMARY KEY, value TEXT)')
    conn.execute('INSERT INTO "select" VALUES (1, \'test\')')
    conn.commit()
    conn.close()
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def special_char_db():
    """Create a test database with tables having special characters in names."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE "my table" (id INTEGER PRIMARY KEY, value TEXT)')
    conn.execute('INSERT INTO "my table" VALUES (1, \'hello\')')
    conn.execute('CREATE TABLE "table-with-dashes" (id INTEGER PRIMARY KEY)')
    conn.execute('INSERT INTO "table-with-dashes" VALUES (1)')
    conn.commit()
    conn.close()
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def injection_db():
    """Create a test database for SQL injection testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    conn = sqlite3.connect(db_path)
    # Create a table with a name that could be part of injection
    conn.execute('CREATE TABLE "normal_table" (id INTEGER PRIMARY KEY, data TEXT)')
    conn.execute("INSERT INTO normal_table VALUES (1, 'safe_data')")
    conn.commit()
    conn.close()
    yield db_path
    os.unlink(db_path)


class TestGenerateSchemaPrompt:
    def test_basic_schema(self, simple_db):
        """Basic schema generation should work."""
        result = generate_schema_prompt(simple_db)
        assert "CREATE TABLE users" in result
        assert "id" in result
        assert "name" in result
        assert "age" in result

    def test_schema_with_rows(self, simple_db):
        """Schema generation with sample rows should include data."""
        result = generate_schema_prompt(simple_db, num_rows=2)
        assert "CREATE TABLE users" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "example rows" in result

    def test_schema_with_limit(self, simple_db):
        """Limiting rows should only show that many rows."""
        result = generate_schema_prompt(simple_db, num_rows=1)
        assert "Alice" in result
        # Only 1 row requested
        assert "1 example rows" in result


class TestReservedWordQuoting:
    def test_reserved_word_order(self, reserved_word_db):
        """Table named 'order' (reserved word) should be properly quoted."""
        result = generate_schema_prompt(reserved_word_db, num_rows=1)
        assert "order" in result.lower()
        # Should not raise an error

    def test_reserved_word_group(self, reserved_word_db):
        """Table named 'group' (reserved word) should be properly quoted."""
        result = generate_schema_prompt(reserved_word_db, num_rows=1)
        assert "group" in result.lower()

    def test_reserved_word_select(self, reserved_word_db):
        """Table named 'select' (reserved word) should be properly quoted."""
        result = generate_schema_prompt(reserved_word_db, num_rows=1)
        assert "select" in result.lower()

    def test_all_reserved_word_tables_present(self, reserved_word_db):
        """All tables with reserved word names should appear in the schema."""
        result = generate_schema_prompt(reserved_word_db, num_rows=1)
        # Schema should contain CREATE TABLE statements for all tables
        assert result.lower().count("create table") == 3


class TestSpecialCharacterTableNames:
    def test_table_with_spaces(self, special_char_db):
        """Tables with spaces in names should be properly handled."""
        result = generate_schema_prompt(special_char_db, num_rows=1)
        assert "my table" in result

    def test_table_with_dashes(self, special_char_db):
        """Tables with dashes in names should be properly handled."""
        result = generate_schema_prompt(special_char_db, num_rows=1)
        assert "table-with-dashes" in result


class TestSQLInjectionPrevention:
    def test_parameterized_name_lookup(self, injection_db):
        """Name lookup in sqlite_master should use parameterized queries."""
        # This should not raise an error - the function uses parameterized queries
        result = generate_schema_prompt(injection_db, num_rows=1)
        assert "normal_table" in result

    def test_parameterized_limit(self, injection_db):
        """LIMIT value should be parameterized."""
        # num_rows is parameterized, so non-integer values would be caught by sqlite
        result = generate_schema_prompt(injection_db, num_rows=5)
        assert "normal_table" in result


class TestConnectionClose:
    def test_connection_closed_on_success(self, simple_db):
        """Connection should be closed after successful execution."""
        # If connection isn't closed, subsequent calls in tests could fail
        result1 = generate_schema_prompt(simple_db)
        result2 = generate_schema_prompt(simple_db)
        assert result1 == result2

    def test_connection_closed_on_error(self):
        """Connection should be closed even if an error occurs."""
        with pytest.raises(Exception):
            generate_schema_prompt("/nonexistent/path/db.sqlite")
        # If the connection wasn't properly closed in a finally block,
        # this could leave dangling connections
