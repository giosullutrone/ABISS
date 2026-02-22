"""Extract secondary preferences from ground-truth SQL.

Only extracts information that does NOT leak the semantic core of the query
(i.e., what the query is about).  Safe to expose: ORDER BY, LIMIT, DISTINCT.
NOT safe: WHERE conditions, JOIN conditions, aggregation targets — these
are part of the semantic answer itself.
"""

import re


def extract_secondary_preferences(sql: str | None) -> str | None:
    """Return a natural-language description of secondary preferences from SQL.

    Returns None if no secondary preferences are found or sql is None.
    """
    if sql is None:
        return None

    sql_upper = sql.upper()
    preferences: list[str] = []

    # --- ORDER BY ---
    order_match = re.search(
        r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT\s|\s*;?\s*$)',
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if order_match:
        order_clause = order_match.group(1).strip().rstrip(';').strip()
        preferences.append(f"Results should be ordered by: {order_clause}")

    # --- LIMIT ---
    limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
    if limit_match:
        preferences.append(f"Only the first {limit_match.group(1)} results are needed")

    # --- DISTINCT ---
    if re.search(r'\bSELECT\s+DISTINCT\b', sql_upper):
        preferences.append("Only unique/distinct results are expected")

    if not preferences:
        return None

    return "; ".join(preferences)
