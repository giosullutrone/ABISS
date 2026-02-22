"""Tests for mask_sql_values in validators/duplicate_removal.py"""
import pytest
from validators.duplicate_removal import mask_sql_values


class TestMaskSqlValues:
    def test_single_quoted_strings_masked(self):
        """Single-quoted string literals should be masked."""
        sql = "SELECT name FROM school WHERE name = 'John'"
        result = mask_sql_values(sql)
        assert "'John'" not in result
        assert "[MASK]" in result
        assert "name" in result  # column name preserved

    def test_numeric_literals_masked(self):
        """Numeric literals should be masked."""
        sql = "SELECT name FROM school WHERE age > 18"
        result = mask_sql_values(sql)
        assert "18" not in result
        assert "[MASK]" in result

    def test_float_literals_masked(self):
        """Float literals should be masked."""
        sql = "SELECT * FROM products WHERE price > 19.99"
        result = mask_sql_values(sql)
        assert "19.99" not in result
        assert "[MASK]" in result

    def test_double_quoted_identifiers_not_masked(self):
        """Double-quoted identifiers (SQLite standard) must NOT be masked."""
        sql = 'SELECT "user name" FROM "my table" WHERE "user name" = \'John\''
        result = mask_sql_values(sql)
        # Double-quoted identifiers should be preserved
        assert '"user name"' in result
        assert '"my table"' in result
        # But the string literal 'John' should be masked
        assert "'John'" not in result
        assert "[MASK]" in result

    def test_escaped_single_quotes_in_strings(self):
        """Escaped single quotes within strings should be handled."""
        sql = "SELECT * FROM t WHERE name = 'O''Brien'"
        result = mask_sql_values(sql)
        assert "O''Brien" not in result
        assert "[MASK]" in result

    def test_multiple_values_masked(self):
        """Multiple values in a query should all be masked."""
        sql = "SELECT * FROM t WHERE age > 18 AND name = 'John' AND score = 95.5"
        result = mask_sql_values(sql)
        assert "18" not in result
        assert "'John'" not in result
        assert "95.5" not in result
        assert result.count("[MASK]") == 3

    def test_structure_preserved(self):
        """SQL structure (keywords, identifiers) should remain intact."""
        sql = "SELECT name, age FROM users WHERE age > 18 AND city = 'NYC'"
        result = mask_sql_values(sql)
        assert "SELECT" in result
        assert "name" in result
        assert "age" in result
        assert "FROM" in result
        assert "users" in result
        assert "WHERE" in result
        assert "AND" in result
        assert "city" in result

    def test_no_values_unchanged(self):
        """A query with no literal values should remain unchanged."""
        sql = "SELECT name FROM users ORDER BY name"
        result = mask_sql_values(sql)
        assert result == sql

    def test_same_structure_different_values_produce_same_template(self):
        """Queries with same structure but different values should produce identical templates."""
        sql1 = "SELECT * FROM t WHERE age > 18 AND name = 'John'"
        sql2 = "SELECT * FROM t WHERE age > 25 AND name = 'Jane'"
        assert mask_sql_values(sql1) == mask_sql_values(sql2)

    def test_different_columns_produce_different_templates(self):
        """Queries with different column names should produce different templates."""
        sql1 = "SELECT name FROM users WHERE age > 18"
        sql2 = "SELECT email FROM users WHERE age > 18"
        assert mask_sql_values(sql1) != mask_sql_values(sql2)

    def test_double_quoted_identifiers_different_produce_different_templates(self):
        """Queries with different double-quoted identifiers should produce different templates."""
        sql1 = 'SELECT "first name" FROM users WHERE id = 1'
        sql2 = 'SELECT "last name" FROM users WHERE id = 1'
        result1 = mask_sql_values(sql1)
        result2 = mask_sql_values(sql2)
        assert result1 != result2
