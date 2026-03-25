"""AST-based SQL fragment extraction for user simulator grounding.

Parses GT SQL into addressable clause-level nodes using sqlglot.
Following BIRD-Interact's (Li et al., 2025) AST-based retrieval approach.
"""

from dataclasses import dataclass

import sqlglot
from sqlglot import exp


@dataclass
class SQLNode:
    node_id: int
    clause_type: str
    sql_fragment: str


def parse_sql_to_nodes(sql: str | None) -> list[SQLNode]:
    """Parse GT SQL into clause-level AST nodes.

    Returns an empty list if sql is None or unparseable.
    """
    if sql is None:
        return []

    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return []

    nodes: list[SQLNode] = []
    node_id = 1

    # --- DISTINCT ---
    select_node = tree.find(exp.Select)
    if select_node and select_node.args.get("distinct"):
        nodes.append(SQLNode(node_id=node_id, clause_type="DISTINCT", sql_fragment="DISTINCT"))
        node_id += 1

    # --- SELECT expressions ---
    if select_node:
        for expression in select_node.expressions:
            nodes.append(SQLNode(
                node_id=node_id,
                clause_type="SELECT",
                sql_fragment=expression.sql(dialect="sqlite"),
            ))
            node_id += 1

    # --- JOIN clauses (outermost query only) ---
    for join in (select_node.args.get("joins") or []) if select_node else []:
        nodes.append(SQLNode(
            node_id=node_id,
            clause_type="JOIN",
            sql_fragment=join.sql(dialect="sqlite"),
        ))
        node_id += 1

    # --- WHERE conditions ---
    where = tree.find(exp.Where)
    if where:
        where_expr = where.this
        conditions = _flatten_and(where_expr)
        for cond in conditions:
            fragment = cond.sql(dialect="sqlite")
            if not _is_join_condition(cond):
                nodes.append(SQLNode(
                    node_id=node_id,
                    clause_type="WHERE",
                    sql_fragment=fragment,
                ))
                node_id += 1

    # --- GROUP BY ---
    group = tree.find(exp.Group)
    if group:
        nodes.append(SQLNode(
            node_id=node_id,
            clause_type="GROUP_BY",
            sql_fragment=group.sql(dialect="sqlite").replace("GROUP BY ", ""),
        ))
        node_id += 1

    # --- HAVING ---
    having = tree.find(exp.Having)
    if having:
        nodes.append(SQLNode(
            node_id=node_id,
            clause_type="HAVING",
            sql_fragment=having.this.sql(dialect="sqlite"),
        ))
        node_id += 1

    # --- ORDER BY ---
    order = tree.find(exp.Order)
    if order:
        nodes.append(SQLNode(
            node_id=node_id,
            clause_type="ORDER_BY",
            sql_fragment=order.sql(dialect="sqlite").replace("ORDER BY ", ""),
        ))
        node_id += 1

    # --- LIMIT ---
    limit = tree.find(exp.Limit)
    if limit:
        # sqlglot stores the limit value in .expression (not .this)
        limit_expr = limit.expression or limit.this
        if limit_expr is not None:
            nodes.append(SQLNode(
                node_id=node_id,
                clause_type="LIMIT",
                sql_fragment=limit_expr.sql(dialect="sqlite"),
            ))
            node_id += 1

    return nodes


def format_nodes_for_prompt(nodes: list[SQLNode]) -> str:
    """Render nodes as a numbered list for Stage 1 classifier prompt."""
    if not nodes:
        return ""
    return "\n".join(
        f"[{n.node_id}] {n.clause_type}: {n.sql_fragment}"
        for n in nodes
    )


def _flatten_and(expr: exp.Expression) -> list[exp.Expression]:
    """Flatten nested AND expressions into a list of individual conditions."""
    if isinstance(expr, exp.And):
        return _flatten_and(expr.left) + _flatten_and(expr.right)
    return [expr]


def _is_join_condition(cond: exp.Expression) -> bool:
    """Heuristic: detect implicit join conditions (col = col from different tables)."""
    if isinstance(cond, exp.EQ):
        left = cond.left
        right = cond.right
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            left_table = left.table
            right_table = right.table
            if left_table and right_table and left_table != right_table:
                return True
    return False
