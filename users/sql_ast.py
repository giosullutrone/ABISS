"""AST-based SQL fragment extraction for user simulator grounding.

Parses GT SQL into addressable clause-level nodes using sqlglot.
Following BIRD-Interact's (Li et al., 2025) AST-based retrieval approach.
"""

import re
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

    # --- FROM tables ---
    if select_node:
        from_clause = select_node.args.get("from_")
        if from_clause:
            for table in from_clause.find_all(exp.Table):
                fragment = table.sql(dialect="sqlite")
                if fragment:
                    nodes.append(SQLNode(
                        node_id=node_id,
                        clause_type="FROM",
                        sql_fragment=fragment,
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


def format_nodes_as_nl(all_nodes: list[SQLNode], selected_ids: set[int]) -> list[str]:
    """Convert selected AST nodes to NL descriptions for Stage 2 prompts.

    Uses all_nodes to build a table-alias map so that fragments like
    ``pa.standing_tackle`` render as ``standing_tackle (from Player_Attributes)``
    instead of ``standing_tackle (from pa)``.
    """
    alias_map = _build_alias_map(all_nodes)
    return [
        _node_to_nl(n, alias_map)
        for n in all_nodes
        if n.node_id in selected_ids
    ]


# ---------------------------------------------------------------------------
# NL conversion helpers
# ---------------------------------------------------------------------------

_NL_LABELS: dict[str, str] = {
    "DISTINCT": "Only unique results",
    "SELECT": "You want",
    "FROM": "From the table",
    "JOIN": "Combined with table",
    "WHERE": "Condition",
    "GROUP_BY": "Grouped by",
    "HAVING": "Group condition",
    "ORDER_BY": "Sorted by",
    "LIMIT": "Limited to",
}


def _build_alias_map(nodes: list[SQLNode]) -> dict[str, str]:
    """Build alias -> full table name map from FROM and JOIN nodes."""
    alias_map: dict[str, str] = {}
    for node in nodes:
        if node.clause_type not in ("FROM", "JOIN"):
            continue
        clean = node.sql_fragment.replace('"', '')
        clean = re.sub(
            r'^(INNER |LEFT |RIGHT |CROSS |FULL OUTER )?JOIN\s+',
            '', clean, flags=re.IGNORECASE,
        )
        m = re.match(r'(\w+)\s+AS\s+(\w+)', clean, re.IGNORECASE)
        if m:
            alias_map[m.group(2)] = m.group(1)
    return alias_map


def _resolve_aliases(text: str, alias_map: dict[str, str]) -> str:
    for alias, full_name in alias_map.items():
        text = re.sub(
            rf'\b{re.escape(alias)}\.(\w+)',
            rf'\1 (from {full_name})',
            text,
        )
        text = text.replace(f'(from {alias})', f'(from {full_name})')
    return text


def _extract_column_name(sql_fragment: str) -> str | None:
    """Extract the leaf column name from a SQL fragment for identity anchoring.

    Handles patterns like: ``t.column``, ``column``, ``AGG(t.column)``,
    ``AGG(DISTINCT t.column)``, ``table.column AS alias``.
    Returns the bare column name or None if not a simple column reference.
    """
    clean = sql_fragment.replace('"', '').strip()
    # Strip trailing AS alias
    clean = re.sub(r'\s+AS\s+\w+$', '', clean, flags=re.IGNORECASE)
    # Unwrap one layer of aggregate: AGG(...) or AGG(DISTINCT ...)
    m_agg = re.match(
        r'(?:AVG|COUNT|SUM|MAX|MIN|GROUP_CONCAT)\(\s*(?:DISTINCT\s+)?(.+)\)$',
        clean, re.IGNORECASE,
    )
    if m_agg:
        clean = m_agg.group(1).strip()
    # Match table.column or bare column
    m_col = re.match(r'(?:\w+\.)?(\w+)$', clean)
    if m_col:
        return m_col.group(1)
    return None


def _node_to_nl(node: SQLNode, alias_map: dict[str, str]) -> str:
    if node.clause_type == "DISTINCT":
        return _NL_LABELS["DISTINCT"]

    text = node.sql_fragment
    clause = node.clause_type

    # Extract the raw column name before any NL rewriting, so we can anchor
    # it in the output and prevent the downstream LLM from drifting to a
    # similarly-named but wrong column (e.g. "types" vs "supertypes").
    raw_col = _extract_column_name(text) if clause in ("SELECT", "WHERE", "GROUP_BY", "HAVING", "ORDER_BY") else None

    # Strip SQL quotes
    text = text.replace('"', '')

    # Remove AS aliases in SELECT / FROM / JOIN
    if clause in ("SELECT", "FROM", "JOIN"):
        text = re.sub(r'\s+AS\s+\w+', '', text, flags=re.IGNORECASE)

    # LIKE -> contains (before alias resolution while pattern is intact)
    text = re.sub(
        r"(\w+(?:\.\w+)?)\s+LIKE\s+'%([^%]+)%'",
        r'\1 contains "\2"', text, flags=re.IGNORECASE,
    )

    # IN (...) -> is one of (...)
    text = re.sub(r'\bIN\s*\(', 'is one of (', text, flags=re.IGNORECASE)

    # Resolve table aliases to full names
    text = _resolve_aliases(text, alias_map)

    # Common aggregates
    text = re.sub(r'AVG\(([^)]+)\)', r'the average of \1', text, flags=re.IGNORECASE)
    text = re.sub(r'COUNT\(\*\)', 'the count of all records', text, flags=re.IGNORECASE)
    text = re.sub(r'COUNT\(DISTINCT\s+([^)]+)\)', r'the count of distinct \1', text, flags=re.IGNORECASE)
    text = re.sub(r'COUNT\(([^)]+)\)', r'the count of \1', text, flags=re.IGNORECASE)
    text = re.sub(r'SUM\(([^)]+)\)', r'the sum of \1', text, flags=re.IGNORECASE)
    text = re.sub(r'MAX\(([^)]+)\)', r'the maximum \1', text, flags=re.IGNORECASE)
    text = re.sub(r'MIN\(([^)]+)\)', r'the minimum \1', text, flags=re.IGNORECASE)

    # STRFTIME
    text = re.sub(r"STRFTIME\('%Y',\s*([^)]+)\)", r'the year of \1', text, flags=re.IGNORECASE)
    text = re.sub(r"STRFTIME\('%m',\s*([^)]+)\)", r'the month of \1', text, flags=re.IGNORECASE)

    # ORDER direction
    text = re.sub(r'\bDESC\b', '(descending)', text)
    text = re.sub(r'\bASC\b', '(ascending)', text)

    # Remaining unresolved alias.column
    text = re.sub(r'(\w+)\.(\w+)', r'\2 (from \1)', text)

    # Clean up JOIN: strip keyword prefix and ON condition
    if clause == "JOIN":
        text = re.sub(r'^(INNER |LEFT |RIGHT |CROSS |FULL OUTER )?JOIN\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+ON\s+.+$', '', text, flags=re.IGNORECASE)

    label = _NL_LABELS.get(clause, clause)
    nl = f"{label}: {text.strip()}"

    # Append column identity anchor for SELECT/WHERE/ORDER_BY nodes so the
    # downstream LLM cannot drift to a similarly-named column.
    # e.g. "You want: types (from cards) [column: types]"
    if raw_col and raw_col.lower() not in nl.lower().split('[')[0]:
        # Column name was lost during NL rewriting; re-anchor it.
        nl += f" [column: {raw_col}]"
    elif raw_col:
        # Column name still present but add anchor for clarity on ambiguous names
        # (e.g., "types" could be confused with "supertypes" without the anchor)
        nl += f" [column: {raw_col}]"

    return nl


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
