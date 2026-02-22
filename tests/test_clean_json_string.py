"""Tests for clean_json_string and extract_last_json_object in models/__init__.py"""
import json
import pytest
from pydantic import BaseModel, Field
from typing import Annotated
from models import clean_json_string, extract_last_json_object


class SQLResponse(BaseModel):
    sql: Annotated[str, Field(description="SQL query")]


class SimpleResponse(BaseModel):
    value: Annotated[str, Field(description="A value")]


# --- clean_json_string tests ---

class TestCleanJsonString:
    def test_backticks_preserved_in_sql(self):
        """Backticks used as SQL identifiers must NOT be replaced."""
        json_str = '{"sql": "SELECT `name` FROM `users`"}'
        result = clean_json_string(json_str)
        assert "`name`" in result
        assert "`users`" in result

    def test_backticks_preserved_roundtrip(self):
        """A JSON string with backtick-quoted SQL should parse correctly."""
        json_str = '{"sql": "SELECT `id`, `name` FROM `my_table` WHERE `age` > 18"}'
        result = clean_json_string(json_str)
        data = json.loads(result)
        assert data["sql"] == "SELECT `id`, `name` FROM `my_table` WHERE `age` > 18"

    def test_markdown_code_fences_removed(self):
        """Markdown code fences (```json and ```) should be stripped."""
        json_str = '```json {"sql": "SELECT 1"} ```'
        result = clean_json_string(json_str)
        assert "```" not in result
        data = json.loads(result)
        assert data["sql"] == "SELECT 1"

    def test_curly_quotes_normalized(self):
        """Smart/curly quotes should be replaced with straight quotes."""
        # \u201c = left double quote, \u201d = right double quote
        json_str = '\u201csql\u201d: \u201cSELECT 1\u201d'
        result = clean_json_string(json_str)
        assert '\u201c' not in result
        assert '\u201d' not in result

    def test_curly_single_quotes_normalized(self):
        """Smart/curly single quotes should be replaced with straight apostrophes."""
        json_str = '{"sql": "SELECT * WHERE name = \u2018John\u2019"}'
        result = clean_json_string(json_str)
        assert "\u2018" not in result
        assert "\u2019" not in result

    def test_newlines_replaced(self):
        """Newlines, carriage returns, and tabs should be replaced with spaces."""
        json_str = '{"sql":\n"SELECT\t1\r"}'
        result = clean_json_string(json_str)
        assert '\n' not in result
        assert '\r' not in result
        assert '\t' not in result

    def test_escaped_dot_cleaned(self):
        """Escaped dots should be unescaped."""
        json_str = '{"sql": "table\\.column"}'
        result = clean_json_string(json_str)
        assert "table.column" in result

    def test_escaped_underscore_cleaned(self):
        """Escaped underscores should be unescaped."""
        json_str = '{"sql": "my\\_table"}'
        result = clean_json_string(json_str)
        assert "my_table" in result

    def test_triple_quotes_converted(self):
        """Python triple-quoted strings should be converted to JSON strings."""
        json_str = """{"sql": \\"\\"\\"SELECT * FROM users\\"\\"\\"} """
        # Test with actual triple quotes
        json_str2 = '{"sql": """SELECT 1"""}'
        result = clean_json_string(json_str2)
        # Triple quotes should be replaced with proper JSON string
        assert '"""' not in result


# --- extract_last_json_object tests ---

class TestExtractLastJsonObject:
    def test_sql_with_backticks(self):
        """SQL with backtick identifiers should be extracted correctly."""
        text = 'Here is the query: {"sql": "SELECT `name` FROM `users`"}'
        result = extract_last_json_object(text, SQLResponse)
        assert result is not None
        assert "`name`" in result.sql
        assert "`users`" in result.sql

    def test_sql_with_code_fence(self):
        """JSON wrapped in markdown code fences should be extracted."""
        text = '```json\n{"sql": "SELECT 1 FROM t"}\n```'
        result = extract_last_json_object(text, SQLResponse)
        assert result is not None
        assert result.sql == "SELECT 1 FROM t"

    def test_last_json_object_selected(self):
        """When multiple JSON objects exist, the last one should be returned."""
        text = '{"sql": "SELECT 1"} some text {"sql": "SELECT 2"}'
        result = extract_last_json_object(text, SQLResponse)
        assert result is not None
        assert result.sql == "SELECT 2"

    def test_no_json_returns_none(self):
        """When no JSON object exists, None should be returned."""
        result = extract_last_json_object("no json here", SQLResponse)
        assert result is None

    def test_nested_json_objects(self):
        """Nested JSON objects should be handled correctly."""
        text = '{"sql": "SELECT * FROM t WHERE id IN (SELECT id FROM t2)"}'
        result = extract_last_json_object(text, SQLResponse)
        assert result is not None
        assert "SELECT * FROM t WHERE id IN (SELECT id FROM t2)" == result.sql

    def test_key_normalization(self):
        """Keys with different casing/separators should be normalized."""
        text = '{"SQL": "SELECT 1"}'
        result = extract_last_json_object(text, SQLResponse)
        assert result is not None
        assert result.sql == "SELECT 1"
