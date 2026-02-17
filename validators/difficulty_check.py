import re
from validators.validator import Validator
from dataset_dataclasses.question import Question, QuestionDifficulty


def classify_sql_difficulty(sql: str) -> QuestionDifficulty:
    """
    Classify the difficulty of a SQL query based on keyword/pattern analysis.
    
    The classification follows the difficulty criteria:
    - SIMPLE: Single table, basic WHERE/ORDER BY, basic aggregates without GROUP BY
    - MODERATE: JOINs, subqueries, GROUP BY + aggregates, complex WHERE (IN/BETWEEN/LIKE), HAVING
    - COMPLEX: Nested subqueries, self-joins, window functions, CTEs, complex grouping
    - HIGHLY_COMPLEX: Multiple/recursive CTEs, UNION, extensive window functions, PIVOT, lateral joins
    """
    if not sql:
        return QuestionDifficulty.SIMPLE

    sql_upper = sql.upper()
    # Normalize whitespace for pattern matching
    sql_normalized = re.sub(r'\s+', ' ', sql_upper).strip()

    # --- Detect SQL features ---

    # CTEs
    cte_matches = re.findall(r'\bWITH\b', sql_normalized)
    # Count CTE definitions (WITH ... AS (...), ... AS (...))
    cte_definitions = len(re.findall(r'\bAS\s*\(', sql_normalized))
    # But only count if there's a WITH keyword before them
    has_cte = len(cte_matches) > 0 and cte_definitions > 0
    multiple_ctes = has_cte and cte_definitions >= 2
    has_recursive_cte = bool(re.search(r'\bWITH\s+RECURSIVE\b', sql_normalized))

    # JOINs
    join_pattern = r'\b(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|CROSS\s+JOIN|LEFT\s+OUTER\s+JOIN|RIGHT\s+OUTER\s+JOIN|FULL\s+OUTER\s+JOIN|NATURAL\s+JOIN|JOIN)\b'
    join_matches = re.findall(join_pattern, sql_normalized)
    join_count = len(join_matches)
    has_join = join_count > 0

    # Self-join detection: same table appearing in multiple FROM/JOIN clauses
    # at the same query level. Strip subqueries first to avoid false positives
    # where the same table is used in outer query and subquery.
    sql_no_subqueries = re.sub(r'\(\s*SELECT\b[^()]*(?:\([^()]*\)[^()]*)*\)', '', sql_normalized)
    table_refs = re.findall(r'\b(?:FROM|JOIN)\s+(\w+)\b', sql_no_subqueries)
    has_self_join = len(table_refs) != len(set(table_refs))

    # Subqueries: SELECT within parentheses (not CTE definitions)
    # Remove CTE definitions first for counting nested subqueries
    sql_no_cte = re.sub(r'\bWITH\b.*?\)\s*(?=SELECT)', '', sql_normalized, flags=re.DOTALL) if has_cte else sql_normalized
    subquery_matches = re.findall(r'\(\s*SELECT\b', sql_no_cte)
    subquery_count = len(subquery_matches)
    has_subquery = subquery_count > 0
    has_nested_subquery = subquery_count >= 2

    # Window functions
    window_functions = [r'\bROW_NUMBER\s*\(', r'\bRANK\s*\(', r'\bDENSE_RANK\s*\(', 
                        r'\bLAG\s*\(', r'\bLEAD\s*\(', r'\bNTILE\s*\(', 
                        r'\bFIRST_VALUE\s*\(', r'\bLAST_VALUE\s*\(', r'\bNTH_VALUE\s*\(']
    window_function_count = sum(1 for wf in window_functions if re.search(wf, sql_normalized))
    has_window_function = window_function_count > 0 or bool(re.search(r'\bOVER\s*\(', sql_normalized))
    extensive_window_functions = window_function_count >= 2

    # Aggregate functions
    aggregate_functions = [r'\bCOUNT\s*\(', r'\bSUM\s*\(', r'\bAVG\s*\(', r'\bMIN\s*\(', r'\bMAX\s*\(']
    aggregate_count = sum(1 for af in aggregate_functions if re.search(af, sql_normalized))
    has_aggregate = aggregate_count > 0
    multiple_aggregates = aggregate_count >= 2

    # GROUP BY
    has_group_by = bool(re.search(r'\bGROUP\s+BY\b', sql_normalized))

    # HAVING
    has_having = bool(re.search(r'\bHAVING\b', sql_normalized))

    # Complex WHERE conditions
    has_in = bool(re.search(r'\bIN\s*\(', sql_normalized))
    has_between = bool(re.search(r'\bBETWEEN\b', sql_normalized))
    has_like = bool(re.search(r'\bLIKE\b', sql_normalized))
    has_complex_where = has_in or has_between or has_like

    # Multiple AND/OR conditions in WHERE
    where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|\bUNION\b|$)', sql_normalized, re.DOTALL)
    where_logical_ops = 0
    if where_match:
        where_clause = where_match.group(1)
        where_logical_ops = len(re.findall(r'\b(AND|OR)\b', where_clause))
    has_multiple_conditions = where_logical_ops >= 2

    # UNION / UNION ALL
    has_union = bool(re.search(r'\bUNION\b', sql_normalized))

    # Advanced features
    has_pivot = bool(re.search(r'\bPIVOT\b', sql_normalized))
    has_lateral = bool(re.search(r'\bLATERAL\b', sql_normalized))
    has_array_ops = bool(re.search(r'\bARRAY\b|UNNEST\s*\(', sql_normalized))

    # Basic features
    has_where = bool(re.search(r'\bWHERE\b', sql_normalized))
    has_order_by = bool(re.search(r'\bORDER\s+BY\b', sql_normalized))

    # Count tables in FROM clause (for single-table detection)
    # Simple heuristic: count distinct table names after FROM (before WHERE/JOIN/GROUP/ORDER)
    from_match = re.search(r'\bFROM\s+(\w+)', sql_normalized)
    
    # --- Classification Logic ---

    # HIGHLY_COMPLEX: Multiple/recursive CTEs, UNION, extensive window functions, 
    # PIVOT, lateral joins, array operations
    highly_complex_indicators = 0
    if multiple_ctes:
        highly_complex_indicators += 1
    if has_recursive_cte:
        highly_complex_indicators += 1
    if has_union:
        highly_complex_indicators += 1
    if extensive_window_functions:
        highly_complex_indicators += 1
    if has_pivot or has_lateral or has_array_ops:
        highly_complex_indicators += 1
    # Also highly complex if combining many complex features
    # Use has_subquery from original SQL (not CTE-stripped) for this combo check
    all_subqueries = re.findall(r'\(\s*SELECT\b', sql_normalized)
    if has_cte and has_window_function and len(all_subqueries) > 0:
        highly_complex_indicators += 2
    # Multiple CTEs with any other complex feature (window function, subquery, multiple joins)
    if multiple_ctes and (has_window_function or len(all_subqueries) > 0 or join_count >= 3):
        highly_complex_indicators += 1

    if highly_complex_indicators >= 2 or has_recursive_cte:
        return QuestionDifficulty.HIGHLY_COMPLEX

    # COMPLEX: Nested subqueries, self-joins, window functions, CTEs, complex grouping
    complex_indicators = 0
    if has_nested_subquery:
        complex_indicators += 1
    if has_self_join:
        complex_indicators += 1
    if has_window_function:
        complex_indicators += 1
    if has_cte:
        complex_indicators += 1
    if multiple_aggregates and has_group_by and has_having:
        complex_indicators += 1
    if has_multiple_conditions and has_having:
        complex_indicators += 1

    if complex_indicators >= 1:
        return QuestionDifficulty.COMPLEX

    # MODERATE: JOINs, subqueries, GROUP BY + aggregates, complex WHERE, HAVING
    moderate_indicators = 0
    if has_join:
        moderate_indicators += 1
    if has_subquery:
        moderate_indicators += 1
    if has_aggregate and has_group_by:
        moderate_indicators += 1
    if has_complex_where:
        moderate_indicators += 1
    if has_having:
        moderate_indicators += 1
    if has_multiple_conditions:
        moderate_indicators += 1
    if multiple_aggregates:
        moderate_indicators += 1

    if moderate_indicators >= 1:
        return QuestionDifficulty.MODERATE

    # SIMPLE: Single table, basic WHERE/ORDER BY, basic aggregates without GROUP BY
    return QuestionDifficulty.SIMPLE


class DifficultyCheck(Validator):
    """
    Automated difficulty validator that checks whether a question's SQL query
    complexity matches its specified difficulty level using keyword/pattern analysis.
    
    No LLM is needed — this is a purely rule-based check.
    For unanswerable questions without SQL, validation is skipped (assumed valid).
    """

    def validate(self, questions: list[Question]) -> list[bool]:
        results: list[bool] = []
        for question in questions:
            if question.sql is None:
                # Unanswerable questions without SQL — cannot verify difficulty 
                # automatically, so we consider them valid
                results.append(True)
                continue
            
            detected_difficulty = classify_sql_difficulty(question.sql)
            results.append(detected_difficulty == question.question_difficulty)
        
        return results
