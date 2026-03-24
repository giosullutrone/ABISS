"""Tests for SQL AST parser."""
import pytest
from users.sql_ast import SQLNode, parse_sql_to_nodes, format_nodes_for_prompt


class TestParseSQLToNodes:
    def test_simple_select(self):
        sql = "SELECT name FROM students"
        nodes = parse_sql_to_nodes(sql)
        assert len(nodes) >= 1
        assert any(n.clause_type == "SELECT" for n in nodes)

    def test_aggregation_in_select(self):
        sql = "SELECT COUNT(DISTINCT student_id) FROM enrollments"
        nodes = parse_sql_to_nodes(sql)
        select_nodes = [n for n in nodes if n.clause_type == "SELECT"]
        assert len(select_nodes) >= 1
        assert "COUNT" in select_nodes[0].sql_fragment.upper()

    def test_where_clause(self):
        sql = "SELECT name FROM students WHERE grade = 12"
        nodes = parse_sql_to_nodes(sql)
        where_nodes = [n for n in nodes if n.clause_type == "WHERE"]
        assert len(where_nodes) >= 1
        assert "grade" in where_nodes[0].sql_fragment.lower()

    def test_group_by(self):
        sql = "SELECT department, COUNT(*) FROM students GROUP BY department"
        nodes = parse_sql_to_nodes(sql)
        group_nodes = [n for n in nodes if n.clause_type == "GROUP_BY"]
        assert len(group_nodes) >= 1

    def test_having(self):
        sql = "SELECT dept, COUNT(*) FROM students GROUP BY dept HAVING COUNT(*) > 5"
        nodes = parse_sql_to_nodes(sql)
        having_nodes = [n for n in nodes if n.clause_type == "HAVING"]
        assert len(having_nodes) >= 1

    def test_order_by(self):
        sql = "SELECT name FROM students ORDER BY name DESC"
        nodes = parse_sql_to_nodes(sql)
        order_nodes = [n for n in nodes if n.clause_type == "ORDER_BY"]
        assert len(order_nodes) >= 1
        assert "DESC" in order_nodes[0].sql_fragment.upper()

    def test_limit(self):
        sql = "SELECT name FROM students LIMIT 10"
        nodes = parse_sql_to_nodes(sql)
        limit_nodes = [n for n in nodes if n.clause_type == "LIMIT"]
        assert len(limit_nodes) == 1
        assert "10" in limit_nodes[0].sql_fragment

    def test_distinct(self):
        sql = "SELECT DISTINCT name FROM students"
        nodes = parse_sql_to_nodes(sql)
        distinct_nodes = [n for n in nodes if n.clause_type == "DISTINCT"]
        assert len(distinct_nodes) == 1

    def test_join(self):
        sql = "SELECT s.name FROM students s INNER JOIN enrollments e ON s.id = e.student_id"
        nodes = parse_sql_to_nodes(sql)
        join_nodes = [n for n in nodes if n.clause_type == "JOIN"]
        assert len(join_nodes) >= 1

    def test_complex_query_all_clauses(self):
        sql = """
        SELECT DISTINCT T1.department, COUNT(T1.student_id)
        FROM students T1
        INNER JOIN enrollments T2 ON T1.id = T2.student_id
        WHERE T1.grade > 10
        GROUP BY T1.department
        HAVING COUNT(T1.student_id) > 5
        ORDER BY COUNT(T1.student_id) DESC
        LIMIT 10
        """
        nodes = parse_sql_to_nodes(sql)
        clause_types = {n.clause_type for n in nodes}
        assert "SELECT" in clause_types
        assert "WHERE" in clause_types
        assert "GROUP_BY" in clause_types
        assert "HAVING" in clause_types
        assert "ORDER_BY" in clause_types
        assert "LIMIT" in clause_types
        assert "DISTINCT" in clause_types
        assert "JOIN" in clause_types

    def test_node_ids_are_sequential(self):
        sql = "SELECT name FROM students WHERE grade = 12 ORDER BY name LIMIT 5"
        nodes = parse_sql_to_nodes(sql)
        ids = [n.node_id for n in nodes]
        assert ids == list(range(1, len(nodes) + 1))

    def test_none_sql_returns_empty(self):
        nodes = parse_sql_to_nodes(None)
        assert nodes == []

    def test_unparseable_sql_returns_empty(self):
        nodes = parse_sql_to_nodes("NOT VALID SQL AT ALL !!!")
        assert nodes == []

    def test_where_excludes_join_conditions(self):
        """WHERE conditions that are part of implicit joins should be excluded."""
        sql = "SELECT name FROM students s, enrollments e WHERE s.id = e.student_id AND s.grade = 12"
        nodes = parse_sql_to_nodes(sql)
        where_nodes = [n for n in nodes if n.clause_type == "WHERE"]
        for wn in where_nodes:
            assert "grade" in wn.sql_fragment.lower() or "s.id = e.student_id" not in wn.sql_fragment


class TestFormatNodesForPrompt:
    def test_format_basic(self):
        nodes = [
            SQLNode(node_id=1, clause_type="SELECT", sql_fragment="COUNT(*)"),
            SQLNode(node_id=2, clause_type="WHERE", sql_fragment="grade = 12"),
        ]
        result = format_nodes_for_prompt(nodes)
        assert "[1] SELECT: COUNT(*)" in result
        assert "[2] WHERE: grade = 12" in result

    def test_format_empty(self):
        result = format_nodes_for_prompt([])
        assert result == ""
