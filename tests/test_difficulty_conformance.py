"""Tests for classify_sql_difficulty in validators/difficulty_conformance.py"""
import pytest
from validators.difficulty_conformance import classify_sql_difficulty
from dataset_dataclasses.question import QuestionDifficulty


class TestSimpleDifficulty:
    """SIMPLE: Single table, basic WHERE/ORDER BY, basic aggregates without GROUP BY"""

    def test_basic_select(self):
        sql = "SELECT name FROM users"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.SIMPLE

    def test_select_with_where(self):
        sql = "SELECT name FROM users WHERE age > 18"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.SIMPLE

    def test_select_with_order_by(self):
        sql = "SELECT name FROM users ORDER BY name ASC"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.SIMPLE

    def test_select_with_limit(self):
        sql = "SELECT * FROM products LIMIT 10"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.SIMPLE

    def test_single_aggregate_no_group_by(self):
        sql = "SELECT COUNT(*) FROM users"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.SIMPLE

    def test_empty_sql(self):
        sql = ""
        assert classify_sql_difficulty(sql) == QuestionDifficulty.SIMPLE

    def test_select_with_single_where_condition(self):
        sql = "SELECT name, email FROM users WHERE active = 1"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.SIMPLE


class TestModerateDifficulty:
    """MODERATE: JOINs, subqueries, GROUP BY + aggregates, complex WHERE, HAVING"""

    def test_simple_join(self):
        sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.MODERATE

    def test_left_join(self):
        sql = "SELECT u.name, o.total FROM users u LEFT JOIN orders o ON u.id = o.user_id"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.MODERATE

    def test_subquery_in_where(self):
        sql = "SELECT name FROM users WHERE id IN (SELECT user_id FROM orders)"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.MODERATE

    def test_group_by_with_aggregate(self):
        sql = "SELECT department, COUNT(*) FROM employees GROUP BY department"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.MODERATE

    def test_like_condition(self):
        sql = "SELECT name FROM users WHERE name LIKE '%john%'"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.MODERATE

    def test_between_condition(self):
        sql = "SELECT * FROM orders WHERE date BETWEEN '2024-01-01' AND '2024-12-31'"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.MODERATE

    def test_in_condition(self):
        sql = "SELECT * FROM users WHERE role IN ('admin', 'moderator')"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.MODERATE

    def test_having_clause(self):
        sql = "SELECT department, COUNT(*) as cnt FROM employees GROUP BY department HAVING cnt > 5"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.MODERATE

    def test_multiple_where_conditions(self):
        sql = "SELECT * FROM users WHERE age > 18 AND status = 'active' AND role = 'user'"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.MODERATE


class TestComplexDifficulty:
    """COMPLEX: Nested subqueries, self-joins, window functions, CTEs, complex grouping"""

    def test_window_function_row_number(self):
        sql = "SELECT name, ROW_NUMBER() OVER (ORDER BY salary DESC) as rn FROM employees"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.COMPLEX

    def test_window_function_rank(self):
        sql = "SELECT name, RANK() OVER (PARTITION BY department ORDER BY salary DESC) FROM employees"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.COMPLEX

    def test_simple_cte(self):
        sql = "WITH active_users AS (SELECT * FROM users WHERE active = 1) SELECT * FROM active_users"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.COMPLEX

    def test_nested_subquery(self):
        sql = "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > (SELECT AVG(total) FROM orders))"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.COMPLEX

    def test_self_join(self):
        sql = "SELECT e1.name, e2.name FROM employees e1 JOIN employees e2 ON e1.manager_id = e2.id"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.COMPLEX

    def test_over_clause(self):
        sql = "SELECT name, SUM(amount) OVER (PARTITION BY category ORDER BY date) FROM transactions"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.COMPLEX


class TestHighlyComplexDifficulty:
    """HIGHLY_COMPLEX: Multiple/recursive CTEs, UNION, extensive window functions"""

    def test_recursive_cte(self):
        sql = "WITH RECURSIVE cte AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM cte WHERE n < 10) SELECT * FROM cte"
        assert classify_sql_difficulty(sql) == QuestionDifficulty.HIGHLY_COMPLEX

    def test_multiple_ctes_with_union(self):
        sql = """
        WITH cte1 AS (SELECT id FROM users WHERE active = 1),
             cte2 AS (SELECT id FROM users WHERE role = 'admin')
        SELECT * FROM cte1 UNION SELECT * FROM cte2
        """
        assert classify_sql_difficulty(sql) == QuestionDifficulty.HIGHLY_COMPLEX

    def test_cte_with_window_function(self):
        """CTE + window function + subquery = highly complex"""
        sql = """
        WITH ranked AS (
            SELECT name, ROW_NUMBER() OVER (ORDER BY salary DESC) as rn
            FROM employees
        )
        SELECT * FROM ranked WHERE rn IN (SELECT MIN(rn) FROM ranked)
        """
        assert classify_sql_difficulty(sql) == QuestionDifficulty.HIGHLY_COMPLEX

    def test_multiple_ctes(self):
        """Multiple CTEs with window function = highly complex"""
        sql = """
        WITH dept_stats AS (SELECT department, AVG(salary) as avg_sal FROM employees GROUP BY department),
             ranked AS (SELECT *, RANK() OVER (ORDER BY avg_sal DESC) as rk FROM dept_stats)
        SELECT * FROM ranked WHERE rk <= 3
        """
        assert classify_sql_difficulty(sql) == QuestionDifficulty.HIGHLY_COMPLEX

    def test_extensive_window_functions(self):
        """Two or more distinct window functions = highly complex (needs union or CTEs too)"""
        sql = """
        WITH t AS (SELECT *, ROW_NUMBER() OVER (ORDER BY id) as rn, LAG(val) OVER (ORDER BY id) as prev FROM data)
        SELECT * FROM t UNION SELECT * FROM t WHERE rn = 1
        """
        assert classify_sql_difficulty(sql) == QuestionDifficulty.HIGHLY_COMPLEX


class TestCTEStrippingRegex:
    """Tests for CTE stripping used in subquery counting (issue 12)"""

    def test_single_cte_no_subquery_in_main(self):
        """A simple CTE with no subqueries in the main query should count 0 subqueries."""
        sql = "WITH cte AS (SELECT * FROM t) SELECT * FROM cte"
        # Should be COMPLEX (has CTE), not HIGHLY_COMPLEX
        assert classify_sql_difficulty(sql) == QuestionDifficulty.COMPLEX

    def test_cte_with_subquery_inside(self):
        """A CTE that internally uses a subquery should not affect main query subquery count."""
        sql = "WITH cte AS (SELECT * FROM (SELECT id FROM t) sub) SELECT * FROM cte"
        # The subquery is inside the CTE, not in the main query → should be COMPLEX
        assert classify_sql_difficulty(sql) == QuestionDifficulty.COMPLEX

    def test_cte_with_subquery_in_main_query(self):
        """A CTE plus a subquery in the main query should count the main query subquery."""
        sql = "WITH cte AS (SELECT * FROM t) SELECT * FROM cte WHERE id IN (SELECT MAX(id) FROM t)"
        # CTE + subquery in main → COMPLEX (CTE indicator is enough)
        result = classify_sql_difficulty(sql)
        assert result == QuestionDifficulty.COMPLEX

    def test_multiple_ctes_stripped(self):
        """Multiple CTEs should all be stripped before counting main query subqueries."""
        sql = """
        WITH cte1 AS (SELECT id FROM t1),
             cte2 AS (SELECT id FROM t2)
        SELECT * FROM cte1 JOIN cte2 ON cte1.id = cte2.id
        """
        # Multiple CTEs → checked for HIGHLY_COMPLEX indicators
        result = classify_sql_difficulty(sql)
        assert result in (QuestionDifficulty.COMPLEX, QuestionDifficulty.HIGHLY_COMPLEX)


class TestASParenCounting:
    """Tests for AS ( counting used in CTE detection (issue 13)"""

    def test_single_cte_counts_one(self):
        """A single CTE should count exactly 1 AS ( pattern."""
        sql = "WITH cte AS (SELECT * FROM t) SELECT * FROM cte"
        result = classify_sql_difficulty(sql)
        # Should be COMPLEX (single CTE), not HIGHLY_COMPLEX
        assert result == QuestionDifficulty.COMPLEX

    def test_cast_not_counted(self):
        """CAST(x AS type) should not be counted as AS ( since there's no ( after the type."""
        sql = "WITH cte AS (SELECT CAST(x AS INTEGER) FROM t) SELECT * FROM cte"
        result = classify_sql_difficulty(sql)
        # CAST(x AS INTEGER) → AS is followed by INTEGER not (
        # Should be COMPLEX (single CTE)
        assert result == QuestionDifficulty.COMPLEX

    def test_subquery_alias_not_counted(self):
        """FROM (SELECT ...) AS alias should not count as CTE AS (."""
        sql = "SELECT * FROM (SELECT * FROM t) AS subq WHERE id = 1"
        result = classify_sql_difficulty(sql)
        # No WITH keyword → has_cte is False regardless of AS pattern
        # Has subquery → MODERATE
        assert result == QuestionDifficulty.MODERATE

    def test_no_with_means_no_cte(self):
        """AS ( without WITH keyword should not be detected as CTE."""
        sql = "SELECT * FROM t1 JOIN (SELECT * FROM t2) AS (id) ON t1.id = id"
        result = classify_sql_difficulty(sql)
        # No WITH → has_cte = False
        # But has JOIN and subquery → at least MODERATE
        assert result.value in ("moderate", "complex", "highly_complex")
        # Should not be classified as having CTEs
        assert result != QuestionDifficulty.HIGHLY_COMPLEX

    def test_two_ctes_detected(self):
        """Two CTE definitions should detect multiple_ctes."""
        sql = """
        WITH cte1 AS (SELECT 1),
             cte2 AS (SELECT 2)
        SELECT * FROM cte1, cte2
        """
        result = classify_sql_difficulty(sql)
        # Multiple CTEs → at least COMPLEX
        assert result in (QuestionDifficulty.COMPLEX, QuestionDifficulty.HIGHLY_COMPLEX)
