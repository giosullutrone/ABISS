# Two-Stage User Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-stage user response pipeline with a two-stage classifier/locator + response generator using AST-based SQL retrieval.

**Architecture:** Stage 1 (Encoder) classifies clarification questions and locates relevant SQL AST nodes. Stage 2 (Decoder) generates natural-language-only responses grounded in the matched source. Council voting aggregates Stage 1 labels; tournament selects best Stage 2 answer.

**Tech Stack:** Python 3.11, sqlglot (new dependency), pydantic, vllm (existing)

**Spec:** `docs/superpowers/specs/2026-03-24-two-stage-user-simulator-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `users/sql_ast.py` | Create | Parse GT SQL into AST nodes via sqlglot |
| `users/prompts/user_classify_prompt.py` | Create | Stage 1 Pydantic models + prompt builders |
| `users/prompts/user_answer_prompt.py` | Create | Stage 2 Pydantic models + prompt builders |
| `users/user_response.py` | Rewrite | Two-stage orchestration flow |
| `users/best_user_answer.py` | Modify | Combinations instead of permutations + randomized ordering |
| `users/prompts/best_user_answer_prompt.py` | Modify | TECHNICAL criteria use matched SQL fragments |
| `users/sql_preferences.py` | Delete | Replaced by sql_ast.py |
| `users/prompts/user_response_prompt.py` | Delete | Replaced by Stage 1 + Stage 2 prompts |
| `requirements.txt` | Modify | Add sqlglot |
| `tests/test_sql_ast.py` | Create | Unit tests for AST parser |
| `tests/test_user_classify_prompt.py` | Create | Unit tests for Stage 1 prompts |
| `tests/test_user_answer_prompt.py` | Create | Unit tests for Stage 2 prompts |
| `tests/test_user_response.py` | Create | Integration tests for two-stage flow |
| `tests/test_best_user_answer.py` | Create | Tests for combinations + randomization |

---

### Task 1: Add sqlglot dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add sqlglot to requirements.txt**

Add `sqlglot` to `requirements.txt` (after `seaborn`, alphabetical):

```
sqlglot
```

- [ ] **Step 2: Install and verify**

Run: `pip install sqlglot`
Expected: Installs successfully.

Run: `python -c "import sqlglot; print(sqlglot.__version__)"`
Expected: Prints a version number.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add sqlglot for SQL AST parsing"
```

---

### Task 2: Create SQL AST parser

**Files:**
- Create: `users/sql_ast.py`
- Create: `tests/test_sql_ast.py`

- [ ] **Step 1: Write failing tests for AST parser**

Create `tests/test_sql_ast.py`:

```python
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
        """WHERE conditions that are part of implicit joins should be excluded
        if there's an explicit JOIN for the same tables."""
        sql = "SELECT name FROM students s, enrollments e WHERE s.id = e.student_id AND s.grade = 12"
        nodes = parse_sql_to_nodes(sql)
        where_nodes = [n for n in nodes if n.clause_type == "WHERE"]
        # Should have the grade filter, not the join condition
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sql_ast.py -v`
Expected: FAIL (import errors, module not found)

- [ ] **Step 3: Implement sql_ast.py**

Create `users/sql_ast.py`:

```python
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

    # --- JOIN clauses ---
    for join in tree.find_all(exp.Join):
        nodes.append(SQLNode(
            node_id=node_id,
            clause_type="JOIN",
            sql_fragment=join.sql(dialect="sqlite"),
        ))
        node_id += 1

    # --- WHERE conditions ---
    where = tree.find(exp.Where)
    if where:
        # Extract individual conditions from AND chains
        where_expr = where.this
        conditions = _flatten_and(where_expr)
        for cond in conditions:
            fragment = cond.sql(dialect="sqlite")
            # Skip conditions that look like implicit join conditions
            # (column = column from different tables)
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
        nodes.append(SQLNode(
            node_id=node_id,
            clause_type="LIMIT",
            sql_fragment=limit.this.sql(dialect="sqlite"),
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
        # Both sides are column references with different table prefixes
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            left_table = left.table
            right_table = right.table
            if left_table and right_table and left_table != right_table:
                return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sql_ast.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add users/sql_ast.py tests/test_sql_ast.py
git commit -m "feat: add SQL AST parser with sqlglot for user simulator grounding"
```

---

### Task 3: Create Stage 1 classification prompts

**Files:**
- Create: `users/prompts/user_classify_prompt.py`
- Create: `tests/test_user_classify_prompt.py`

- [ ] **Step 1: Write failing tests for Stage 1 prompts**

Create `tests/test_user_classify_prompt.py`:

```python
"""Tests for Stage 1 classification prompts."""
import pytest
from unittest.mock import MagicMock
from users.prompts.user_classify_prompt import (
    UserClassifySolvable,
    UserClassifyAnswerable,
    get_user_classify_prompt_solvable,
    get_user_classify_prompt_answerable,
    get_classify_solvable_result,
    get_classify_answerable_result,
)
from dataset_dataclasses.benchmark import Conversation, Interaction, SystemResponse, RelevancyLabel, CategoryUse
from dataset_dataclasses.question import Question, QuestionUnanswerable, QuestionStyle, QuestionDifficulty
from users.sql_ast import SQLNode


def _make_category(name="Lexical Vagueness", is_answerable=False, is_solvable=True):
    cat = MagicMock()
    cat.get_name.return_value = name
    cat.is_answerable.return_value = is_answerable
    cat.is_solvable.return_value = is_solvable
    return cat


def _make_solvable_conversation(system_question="What do you mean by 'recent'?"):
    cat = _make_category()
    question = QuestionUnanswerable(
        db_id="test_db",
        question="Show me recent students",
        evidence=None,
        sql="SELECT name FROM students ORDER BY enrollment_date DESC LIMIT 10",
        category=cat,
        question_style=QuestionStyle.COLLOQUIAL,
        question_difficulty=QuestionDifficulty.SIMPLE,
        hidden_knowledge="Recent means enrolled in the last 6 months",
        is_solvable=True,
    )
    conv = Conversation(
        question=question,
        interactions=[Interaction(system_response=SystemResponse(system_question=system_question))],
        category_use=CategoryUse.GROUND_TRUTH,
    )
    return conv


def _make_answerable_conversation(system_question="How many results do you want?"):
    cat = _make_category(name="Answerable", is_answerable=True, is_solvable=True)
    question = Question(
        db_id="test_db",
        question="List all students",
        evidence=None,
        sql="SELECT name FROM students ORDER BY name ASC LIMIT 20",
        category=cat,
        question_style=QuestionStyle.FORMAL,
        question_difficulty=QuestionDifficulty.SIMPLE,
    )
    conv = Conversation(
        question=question,
        interactions=[Interaction(system_response=SystemResponse(system_question=system_question))],
        category_use=CategoryUse.GROUND_TRUTH,
    )
    return conv


class TestSolvablePrompt:
    def test_contains_hidden_knowledge(self):
        conv = _make_solvable_conversation()
        nodes = [SQLNode(1, "ORDER_BY", "enrollment_date DESC"), SQLNode(2, "LIMIT", "10")]
        prompt = get_user_classify_prompt_solvable(conv, nodes)
        assert "Recent means enrolled in the last 6 months" in prompt

    def test_contains_ast_nodes(self):
        conv = _make_solvable_conversation()
        nodes = [SQLNode(1, "ORDER_BY", "enrollment_date DESC"), SQLNode(2, "LIMIT", "10")]
        prompt = get_user_classify_prompt_solvable(conv, nodes)
        assert "[1] ORDER_BY: enrollment_date DESC" in prompt
        assert "[2] LIMIT: 10" in prompt

    def test_contains_clarification_question(self):
        conv = _make_solvable_conversation(system_question="What time period?")
        nodes = []
        prompt = get_user_classify_prompt_solvable(conv, nodes)
        assert "What time period?" in prompt

    def test_contains_all_three_labels(self):
        conv = _make_solvable_conversation()
        prompt = get_user_classify_prompt_solvable(conv, [])
        assert "Relevant" in prompt
        assert "Technical" in prompt
        assert "Irrelevant" in prompt


class TestAnswerablePrompt:
    def test_no_hidden_knowledge(self):
        conv = _make_answerable_conversation()
        nodes = [SQLNode(1, "ORDER_BY", "name ASC"), SQLNode(2, "LIMIT", "20")]
        prompt = get_user_classify_prompt_answerable(conv, nodes)
        assert "Hidden Knowledge" not in prompt

    def test_contains_ast_nodes(self):
        conv = _make_answerable_conversation()
        nodes = [SQLNode(1, "ORDER_BY", "name ASC")]
        prompt = get_user_classify_prompt_answerable(conv, nodes)
        assert "[1] ORDER_BY: name ASC" in prompt

    def test_only_two_labels(self):
        conv = _make_answerable_conversation()
        prompt = get_user_classify_prompt_answerable(conv, [])
        assert "Technical" in prompt
        assert "Irrelevant" in prompt


class TestResultExtraction:
    def test_solvable_relevant(self):
        response = UserClassifySolvable(relevancy="Relevant", node_ids=[])
        label, node_ids = get_classify_solvable_result(response)
        assert label == RelevancyLabel.RELEVANT
        assert node_ids == []

    def test_solvable_technical_with_nodes(self):
        response = UserClassifySolvable(relevancy="Technical", node_ids=[1, 3])
        label, node_ids = get_classify_solvable_result(response)
        assert label == RelevancyLabel.TECHNICAL
        assert node_ids == [1, 3]

    def test_solvable_irrelevant(self):
        response = UserClassifySolvable(relevancy="Irrelevant", node_ids=[])
        label, node_ids = get_classify_solvable_result(response)
        assert label == RelevancyLabel.IRRELEVANT

    def test_answerable_technical(self):
        response = UserClassifyAnswerable(relevancy="Technical", node_ids=[2])
        label, node_ids = get_classify_answerable_result(response)
        assert label == RelevancyLabel.TECHNICAL
        assert node_ids == [2]

    def test_answerable_irrelevant(self):
        response = UserClassifyAnswerable(relevancy="Irrelevant", node_ids=[])
        label, node_ids = get_classify_answerable_result(response)
        assert label == RelevancyLabel.IRRELEVANT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_user_classify_prompt.py -v`
Expected: FAIL (import errors)

- [ ] **Step 3: Implement user_classify_prompt.py**

Create `users/prompts/user_classify_prompt.py`:

```python
"""Stage 1: Classify clarification questions and locate relevant SQL AST nodes.

The classifier sees raw SQL fragments in the AST node list to enable
semantic matching. The NL-only constraint is applied in Stage 2.
"""

from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from utils.prompt_utils import model_field_descriptions, get_conversation_history_prompt
from users.sql_ast import SQLNode, format_nodes_for_prompt


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UserClassifySolvable(BaseModel):
    """For solvable (ambiguous) questions -- all three relevancy labels valid."""
    relevancy: Annotated[
        Literal["Relevant", "Technical", "Irrelevant"],
        Field(description=(
            "Classification of the clarification question: "
            "'Relevant' if it addresses the semantic ambiguity, "
            "'Technical' if it asks about SQL implementation details, "
            "or 'Irrelevant' if it doesn't help or tries to extract SQL. "
            "Put only 'Relevant', 'Technical', or 'Irrelevant'."
        )),
    ]
    node_ids: Annotated[
        list[int],
        Field(description=(
            "If Technical, list ALL SQL node IDs the question relates to. "
            "If Relevant or Irrelevant, leave this as an empty list []."
        )),
    ]


class UserClassifyAnswerable(BaseModel):
    """For answerable questions -- only Technical or Irrelevant."""
    relevancy: Annotated[
        Literal["Technical", "Irrelevant"],
        Field(description=(
            "Classification of the clarification question: "
            "'Technical' if it asks about SQL implementation details, "
            "or 'Irrelevant' if it doesn't help or tries to extract SQL. "
            "Put only 'Technical' or 'Irrelevant'."
        )),
    ]
    node_ids: Annotated[
        list[int],
        Field(description=(
            "If Technical, list ALL SQL node IDs the question relates to. "
            "If Irrelevant, leave this as an empty list []."
        )),
    ]


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def get_classify_solvable_result(response: BaseModel) -> tuple[RelevancyLabel, list[int]]:
    validated = UserClassifySolvable.model_validate(response)
    label_str = validated.relevancy.strip().lower()
    if "irrelevant" in label_str:
        label = RelevancyLabel.IRRELEVANT
    elif "relevant" in label_str:
        label = RelevancyLabel.RELEVANT
    elif "technical" in label_str:
        label = RelevancyLabel.TECHNICAL
    else:
        raise ValueError(f"Invalid relevancy: {validated.relevancy}")
    return label, validated.node_ids


def get_classify_answerable_result(response: BaseModel) -> tuple[RelevancyLabel, list[int]]:
    validated = UserClassifyAnswerable.model_validate(response)
    label_str = validated.relevancy.strip().lower()
    if "technical" in label_str:
        label = RelevancyLabel.TECHNICAL
    elif "irrelevant" in label_str:
        label = RelevancyLabel.IRRELEVANT
    else:
        raise ValueError(f"Invalid relevancy: {validated.relevancy}")
    return label, validated.node_ids


# ---------------------------------------------------------------------------
# Relevancy definitions (shared)
# ---------------------------------------------------------------------------

def _relevancy_definitions(include_relevant: bool) -> str:
    text = ""
    idx = 1
    if include_relevant:
        text += (
            f"**{idx}. RELEVANT** (Addresses semantic ambiguity):\n"
            "- Directly addresses the semantic ambiguity using hidden knowledge\n"
            "- Helps clarify which interpretation the user intends\n"
            "- Focuses on natural language disambiguation (not SQL implementation)\n"
            "- NOT about which columns/tables to use -- about WHICH MEANING is intended\n\n"
        )
        idx += 1

    text += (
        f"**{idx}. TECHNICAL** (Asks about SQL implementation details):\n"
        "- Focuses on output preferences and implementation: ordering, limits, grouping, aggregation\n"
        "- Asks about which columns, tables, or fields to use\n"
        "- Can be answered using the SQL reference nodes listed above\n"
        "- NOT about semantic meaning -- about HOW to implement or present results\n\n"
    )
    idx += 1

    text += (
        f"**{idx}. IRRELEVANT** (Doesn't help resolve the query):\n"
        "- Doesn't address the ambiguity or provide useful technical details\n"
        "- Tries to extract the SQL solution directly\n"
        "- Completely off-topic or tangential to the query\n\n"
    )
    return text


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def get_user_classify_prompt_solvable(conversation: Conversation, nodes: list[SQLNode]) -> str:
    """Stage 1 prompt for solvable (ambiguous) questions."""
    question = conversation.question
    assert conversation.interactions[-1].system_response.system_question is not None

    prompt = (
        "You are a classification engine for a text-to-SQL user simulator. "
        "A text-to-SQL system has asked a clarification question. "
        "Classify it and identify which SQL nodes it relates to.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += "**Question Type:** Solvable -- has ambiguity that can be resolved through clarification\n"

    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Disambiguating Intent):** {question.hidden_knowledge}\n"

    prompt += f"\n**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    nodes_text = format_nodes_for_prompt(nodes)
    if nodes_text:
        prompt += "## SQL Reference Nodes\n"
        prompt += nodes_text + "\n\n"
    else:
        prompt += "## SQL Reference Nodes\nNo SQL nodes available.\n\n"

    prompt += "## Relevancy Definitions\n"
    prompt += _relevancy_definitions(include_relevant=True)

    prompt += "## Classification Rules\n"
    prompt += "- RELEVANT = addresses semantic ambiguity (which meaning?)\n"
    prompt += "- TECHNICAL = asks implementation details (columns, ordering, limits, grouping, aggregation)\n"
    prompt += "- IRRELEVANT = doesn't help or tries to extract SQL\n"
    prompt += "- Questions asking about columns/tables to use are TECHNICAL (not Relevant)\n"
    prompt += "- If TECHNICAL, list ALL node IDs the question relates to\n"
    prompt += "- If RELEVANT or IRRELEVANT, leave node_ids as []\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide brief reasoning (approximately 128 characters), then a JSON object with:\n"
    prompt += model_field_descriptions(UserClassifySolvable) + "\n"

    return prompt


def get_user_classify_prompt_answerable(conversation: Conversation, nodes: list[SQLNode]) -> str:
    """Stage 1 prompt for answerable questions (no semantic ambiguity)."""
    question = conversation.question
    assert conversation.interactions[-1].system_response.system_question is not None

    prompt = (
        "You are a classification engine for a text-to-SQL user simulator. "
        "A text-to-SQL system has asked a clarification question about an already-clear query. "
        "Classify it and identify which SQL nodes it relates to.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += "**Question Type:** Answerable -- question is already clear, no semantic ambiguity\n"

    prompt += f"\n**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    nodes_text = format_nodes_for_prompt(nodes)
    if nodes_text:
        prompt += "## SQL Reference Nodes\n"
        prompt += nodes_text + "\n\n"
    else:
        prompt += "## SQL Reference Nodes\nNo SQL nodes available.\n\n"

    prompt += "## Relevancy Definitions\n"
    prompt += _relevancy_definitions(include_relevant=False)

    prompt += "## Classification Rules\n"
    prompt += "- TECHNICAL = asks implementation details (columns, ordering, limits, grouping, aggregation)\n"
    prompt += "- IRRELEVANT = doesn't help or tries to extract SQL\n"
    prompt += "- Semantic clarification questions are IRRELEVANT (question is already clear)\n"
    prompt += "- If TECHNICAL, list ALL node IDs the question relates to\n"
    prompt += "- If IRRELEVANT, leave node_ids as []\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide brief reasoning (approximately 128 characters), then a JSON object with:\n"
    prompt += model_field_descriptions(UserClassifyAnswerable) + "\n"

    return prompt
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_user_classify_prompt.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add users/prompts/user_classify_prompt.py tests/test_user_classify_prompt.py
git commit -m "feat: add Stage 1 classification prompts for two-stage user simulator"
```

---

### Task 4: Create Stage 2 response generation prompts

**Files:**
- Create: `users/prompts/user_answer_prompt.py`
- Create: `tests/test_user_answer_prompt.py`

- [ ] **Step 1: Write failing tests for Stage 2 prompts**

Create `tests/test_user_answer_prompt.py`:

```python
"""Tests for Stage 2 response generation prompts."""
import pytest
from unittest.mock import MagicMock
from users.prompts.user_answer_prompt import (
    UserAnswerModel,
    get_user_answer_prompt_relevant,
    get_user_answer_prompt_technical,
    get_user_answer_prompt_irrelevant,
)
from dataset_dataclasses.benchmark import Conversation, Interaction, SystemResponse, CategoryUse
from dataset_dataclasses.question import QuestionUnanswerable, QuestionStyle, QuestionDifficulty


def _make_category(is_answerable=False, is_solvable=True):
    cat = MagicMock()
    cat.is_answerable.return_value = is_answerable
    cat.is_solvable.return_value = is_solvable
    return cat


def _make_conversation(hidden_knowledge="Recent means last 6 months", style=QuestionStyle.COLLOQUIAL):
    cat = _make_category()
    question = QuestionUnanswerable(
        db_id="test_db",
        question="Show me recent students",
        evidence=None,
        sql="SELECT name FROM students ORDER BY enrollment_date DESC LIMIT 10",
        category=cat,
        question_style=style,
        question_difficulty=QuestionDifficulty.SIMPLE,
        hidden_knowledge=hidden_knowledge,
        is_solvable=True,
    )
    return Conversation(
        question=question,
        interactions=[Interaction(system_response=SystemResponse(system_question="What do you mean by recent?"))],
        category_use=CategoryUse.GROUND_TRUTH,
    )


class TestRelevantPrompt:
    def test_contains_hidden_knowledge(self):
        conv = _make_conversation()
        prompt = get_user_answer_prompt_relevant(conv)
        assert "Recent means last 6 months" in prompt

    def test_contains_nl_constraint(self):
        conv = _make_conversation()
        prompt = get_user_answer_prompt_relevant(conv)
        assert "natural language" in prompt.lower()
        assert "NEVER" in prompt

    def test_contains_style_info(self):
        conv = _make_conversation(style=QuestionStyle.FORMAL)
        prompt = get_user_answer_prompt_relevant(conv)
        assert "formal" in prompt.lower() or "Formal" in prompt


class TestTechnicalPrompt:
    def test_contains_sql_fragments(self):
        conv = _make_conversation()
        fragments = ["ORDER BY enrollment_date DESC", "LIMIT 10"]
        prompt = get_user_answer_prompt_technical(conv, fragments)
        assert "ORDER BY enrollment_date DESC" in prompt
        assert "LIMIT 10" in prompt

    def test_contains_nl_constraint(self):
        conv = _make_conversation()
        prompt = get_user_answer_prompt_technical(conv, ["LIMIT 10"])
        assert "natural language" in prompt.lower()
        assert "NEVER" in prompt

    def test_empty_fragments_shows_uncertainty_instruction(self):
        conv = _make_conversation()
        prompt = get_user_answer_prompt_technical(conv, [])
        assert "uncertainty" in prompt.lower()

    def test_contains_style_info(self):
        conv = _make_conversation(style=QuestionStyle.COLLOQUIAL)
        prompt = get_user_answer_prompt_technical(conv, ["LIMIT 10"])
        assert "colloquial" in prompt.lower() or "Colloquial" in prompt


class TestIrrelevantPrompt:
    def test_contains_refusal_instruction(self):
        conv = _make_conversation()
        prompt = get_user_answer_prompt_irrelevant(conv)
        assert "refuse" in prompt.lower()

    def test_contains_style_info(self):
        conv = _make_conversation()
        prompt = get_user_answer_prompt_irrelevant(conv)
        assert "style" in prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_user_answer_prompt.py -v`
Expected: FAIL (import errors)

- [ ] **Step 3: Implement user_answer_prompt.py**

Create `users/prompts/user_answer_prompt.py`:

```python
"""Stage 2: Generate natural-language-only responses grounded in matched source.

The NL-only constraint is enforced here. The model receives source material
(hidden_knowledge or SQL fragments) but must paraphrase into natural language
matching the conversation's style.
"""

from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel, Field
from typing import Annotated
from utils.prompt_utils import model_field_descriptions, get_conversation_history_prompt
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES


class UserAnswerModel(BaseModel):
    answer: Annotated[str, Field(description="Your natural language answer to the clarification question.")]


NL_CONSTRAINT = (
    "**CRITICAL CONSTRAINT:** You are a non-technical user. "
    "Respond in natural language ONLY. "
    "NEVER use SQL syntax, column names, table aliases, or query structure. "
    "Express the intent behind the information, not its SQL representation. "
    "The SQL source material is provided as reference for what you want; "
    "your answer must sound like a real non-technical user speaking in their own voice.\n\n"
)


def _style_section(conversation: Conversation) -> str:
    question_style = conversation.question.question_style
    style_desc = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    text = f"**Expected Answer Style (Match Your Original Question):**\n{style_desc}\n"
    text += "Your answer should feel like a natural continuation of your original question.\n\n"
    return text


def get_user_answer_prompt_relevant(conversation: Conversation) -> str:
    """Stage 2 prompt for RELEVANT classification. Source: hidden_knowledge."""
    question = conversation.question
    clarification = conversation.interactions[-1].system_response.system_question

    prompt = (
        "You are a user in a text-to-SQL scenario. "
        "A system asked you a clarification question that addresses the semantic ambiguity in your query. "
        "Answer it using your knowledge of what you actually want.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += f"**Clarification Question:** {clarification}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        prompt += f"## Your Intent\n"
        prompt += f"**What You Actually Mean:** {question.hidden_knowledge}\n\n"

    prompt += NL_CONSTRAINT
    prompt += "## Instructions\n"
    prompt += "Use your intent to disambiguate. Be direct about which interpretation you mean. "
    prompt += "Do NOT mention SQL, databases, or technical implementation.\n\n"

    prompt += _style_section(conversation)

    prompt += "## Response Format\n"
    prompt += "Provide your answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerModel) + "\n"

    return prompt


def get_user_answer_prompt_technical(conversation: Conversation, sql_fragments: list[str]) -> str:
    """Stage 2 prompt for TECHNICAL classification. Source: matched SQL fragments."""
    question = conversation.question
    clarification = conversation.interactions[-1].system_response.system_question

    prompt = (
        "You are a user in a text-to-SQL scenario. "
        "A system asked you a technical question about implementation details. "
        "Answer it based on what you want from the results.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += f"**Clarification Question:** {clarification}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    if sql_fragments:
        prompt += "## What You Want (Reference)\n"
        prompt += "The following describes what you want from the results. "
        prompt += "Use this to inform your answer, but express it in your own words:\n"
        for frag in sql_fragments:
            prompt += f"- {frag}\n"
        prompt += "\n"
    else:
        prompt += "## What You Want (Reference)\n"
        prompt += "No specific preferences defined. Express genuine uncertainty.\n\n"

    prompt += NL_CONSTRAINT
    prompt += "## Instructions\n"
    if sql_fragments:
        prompt += "Express the intent behind the reference information in natural language. "
        prompt += "For example, 'ORDER BY count DESC' should become something like "
        prompt += "'I want the most popular ones first' (colloquial) or "
        prompt += "'Sort by frequency in descending order' (formal). "
        prompt += "Match the style of the original question.\n\n"
    else:
        prompt += "Express genuine uncertainty. "
        prompt += "Examples: 'Either way is fine', 'I'm not sure, whatever makes sense'.\n\n"

    prompt += _style_section(conversation)

    prompt += "## Response Format\n"
    prompt += "Provide your answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerModel) + "\n"

    return prompt


def get_user_answer_prompt_irrelevant(conversation: Conversation) -> str:
    """Stage 2 prompt for IRRELEVANT classification. No source, refuse."""
    question = conversation.question
    clarification = conversation.interactions[-1].system_response.system_question

    prompt = (
        "You are a user in a text-to-SQL scenario. "
        "A system asked you an irrelevant clarification question. "
        "Politely but firmly refuse to answer.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    prompt += f"**Clarification Question:** {clarification}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    prompt += "## Instructions\n"
    prompt += "Politely but firmly refuse to answer. "
    prompt += "Examples: 'That's not relevant to my question', 'I can't answer that', "
    prompt += "'Could you focus on my original question instead?'\n\n"

    prompt += _style_section(conversation)

    prompt += "## Response Format\n"
    prompt += "Provide your answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerModel) + "\n"

    return prompt
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_user_answer_prompt.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add users/prompts/user_answer_prompt.py tests/test_user_answer_prompt.py
git commit -m "feat: add Stage 2 response generation prompts for two-stage user simulator"
```

---

### Task 5: Modify tournament to use combinations + randomized ordering

**Files:**
- Modify: `users/best_user_answer.py`
- Modify: `users/prompts/best_user_answer_prompt.py`
- Create: `tests/test_best_user_answer.py`

- [ ] **Step 1: Write failing tests for combinations + randomization**

Create `tests/test_best_user_answer.py`:

```python
"""Tests for tournament combinations and TECHNICAL prompt changes."""
import pytest
from itertools import combinations
from users.prompts.best_user_answer_prompt import get_best_user_answer_technical_prompt
from unittest.mock import MagicMock
from dataset_dataclasses.benchmark import Conversation, Interaction, SystemResponse, RelevancyLabel, CategoryUse
from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


def _make_conversation():
    cat = MagicMock()
    cat.is_answerable.return_value = True
    cat.is_solvable.return_value = True
    question = Question(
        db_id="test_db",
        question="List students",
        evidence=None,
        sql="SELECT name FROM students ORDER BY name LIMIT 10",
        category=cat,
        question_style=QuestionStyle.FORMAL,
        question_difficulty=QuestionDifficulty.SIMPLE,
    )
    return Conversation(
        question=question,
        interactions=[Interaction(
            system_response=SystemResponse(system_question="How many results?"),
            relevance=RelevancyLabel.TECHNICAL,
        )],
        category_use=CategoryUse.GROUND_TRUTH,
    )


class TestTechnicalPromptWithFragments:
    def test_contains_sql_fragments(self):
        db = MagicMock()
        conv = _make_conversation()
        fragments = ["ORDER BY name ASC", "LIMIT 10"]
        prompt = get_best_user_answer_technical_prompt(db, conv, "Answer A", "Answer B", fragments)
        assert "ORDER BY name ASC" in prompt
        assert "LIMIT 10" in prompt

    def test_contains_intent_criteria(self):
        db = MagicMock()
        conv = _make_conversation()
        prompt = get_best_user_answer_technical_prompt(db, conv, "A", "B", ["LIMIT 10"])
        assert "intent" in prompt.lower()

    def test_empty_fragments_shows_uncertainty(self):
        db = MagicMock()
        conv = _make_conversation()
        prompt = get_best_user_answer_technical_prompt(db, conv, "A", "B", [])
        assert "uncertainty" in prompt.lower()


class TestCombinationsCount:
    def test_three_candidates_produces_three_pairs(self):
        candidates = ["A", "B", "C"]
        pairs = list(combinations(range(len(candidates)), 2))
        assert len(pairs) == 3  # Was 6 with permutations

    def test_two_candidates_produces_one_pair(self):
        candidates = ["A", "B"]
        pairs = list(combinations(range(len(candidates)), 2))
        assert len(pairs) == 1  # Was 2 with permutations
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_best_user_answer.py -v`
Expected: Some tests FAIL (prompt signature changed, fragments parameter)

- [ ] **Step 3: Modify best_user_answer_prompt.py**

In `users/prompts/best_user_answer_prompt.py`, replace the `get_best_user_answer_technical_prompt` function to accept `sql_fragments` instead of using `extract_secondary_preferences`. Update the signature and body:

Replace the import of `extract_secondary_preferences` and the entire `get_best_user_answer_technical_prompt` function with:

```python
def get_best_user_answer_technical_prompt(db: DBDataset,
                                          conversation: Conversation,
                                          generation_a: str,
                                          generation_b: str,
                                          sql_fragments: list[str] | None = None) -> str:
    """Evaluate pairs of answers to TECHNICAL clarification questions.

    Uses matched SQL fragments from AST-based retrieval as the correctness
    reference, rather than the old secondary preferences.
    """
    prompt = _get_best_user_answer_prompt_common(db, conversation, generation_a, generation_b, RelevancyLabel.TECHNICAL)

    if sql_fragments:
        prompt += "\n**Source Information (What the User Wants):**\n"
        for frag in sql_fragments:
            prompt += f"- {frag}\n"
        prompt += "\n"
    else:
        prompt += "\n**Note:** No specific technical preferences defined -- user may express uncertainty.\n"

    prompt += "\n## Candidate Answers\n"
    prompt += f"**Answer A:**\n{generation_a}\n\n"
    prompt += f"**Answer B:**\n{generation_b}\n\n"

    prompt += "## Evaluation Task\n"
    prompt += "Compare the two answers for responding to a TECHNICAL clarification question. "
    prompt += "Both candidates were classified as TECHNICAL. "
    prompt += "Select the one that is more **correct** and, as a tiebreaker, more natural.\n\n"

    if sql_fragments:
        prompt += "**Correctness Criteria (Primary -- decide the winner):**\n"
        prompt += "1. **Intent Accuracy:** Does the answer accurately convey the intent behind the source information in natural language?\n"
        prompt += "2. **No Fabrication:** Does NOT add information beyond what the source material implies.\n"
        prompt += "3. **No Information Leakage:** Avoids revealing SQL syntax, column names, table aliases, or query structure.\n\n"
    else:
        prompt += "**Correctness Criteria (Primary -- decide the winner):**\n"
        prompt += "1. **Appropriate Uncertainty:** Since no preferences are defined, the answer should express genuine uncertainty.\n"
        prompt += "2. **No Fabrication:** Does NOT invent specific preferences.\n"
        prompt += "3. **No Information Leakage:** Avoids revealing SQL details.\n\n"

    prompt += "**Style Criteria (Secondary -- only used to break ties in correctness):**\n"
    prompt += "- **Naturalness:** Sounds like a user stating preferences, not SQL code\n"
    prompt += "- **Style Consistency:** Maintains the style and vocabulary of the original question\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide concise analysis (approximately 256 characters) comparing the answers.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(BestUserAnswerTechnicalResponse) + "\n\n"
    prompt += "In case of a tie, select Answer A."

    return prompt
```

Also remove the `from users.sql_preferences import extract_secondary_preferences` import at line 128 (it's no longer needed).

- [ ] **Step 4: Modify best_user_answer.py for combinations + randomization**

Replace the entire `users/best_user_answer.py` with the following. Key changes from the original:
- `itertools.combinations` instead of nested `for j / for k where j != k` (halves pair count)
- Randomized A/B ordering within each combination to mitigate positional bias
- New `sql_fragments_per_conv` parameter threaded to TECHNICAL prompts
- Removed `cast` import (no longer needed)

```python
from typing import Callable
from itertools import combinations
import random
from pydantic import BaseModel
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.council_tracking import TournamentVotes
from models.model import Model
from db_datasets.db_dataset import DBDataset
from users.prompts.best_user_answer_prompt import (
    get_best_user_answer_relevant_prompt, BestUserAnswerRelevantResponse, get_best_user_answer_relevant_result,
    get_best_user_answer_technical_prompt, BestUserAnswerTechnicalResponse, get_best_user_answer_technical_result,
    get_best_user_answer_irrelevant_prompt, BestUserAnswerIrrelevantResponse, get_best_user_answer_irrelevant_result
)


class BestUserAnswer:
    """Selects the best user answer among candidates via pairwise tournament
    evaluated by the model council.

    Uses combinations (not permutations) with randomized A/B ordering
    to halve comparison count while mitigating positional bias.
    """

    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def select_best_user_answers(
        self,
        conversations: list[Conversation],
        answers: list[list[str]],
        candidate_model_indices: list[list[int]] | None = None,
        sql_fragments_per_conv: list[list[str]] | None = None,
    ) -> tuple[list[str], list[TournamentVotes]]:
        """Run a pairwise tournament for each conversation's candidate answers.

        Uses combinations (A-vs-B once, not A-vs-B and B-vs-A) with
        randomized candidate ordering to mitigate positional bias.
        """
        votes: list[list[int]] = [[0] * len(ans) for ans in answers]

        # Generate pairwise comparison prompts using combinations
        pairwise_prompts: dict[int, list[tuple[int, int, str, type[BaseModel], Callable]]] = {}
        for i, gens in enumerate(answers):
            pairwise_prompts[i] = []
            assert conversations[i].interactions[-1].relevance is not None
            relevance = conversations[i].interactions[-1].relevance

            for j, k in combinations(range(len(gens)), 2):
                # Randomize A/B ordering to mitigate positional bias
                if random.random() < 0.5:
                    a_idx, b_idx = j, k
                else:
                    a_idx, b_idx = k, j

                if relevance == RelevancyLabel.RELEVANT:
                    prompt = get_best_user_answer_relevant_prompt(
                        self.db, conversations[i], gens[a_idx], gens[b_idx])
                    model_class = BestUserAnswerRelevantResponse
                    extractor = get_best_user_answer_relevant_result
                elif relevance == RelevancyLabel.TECHNICAL:
                    frags = sql_fragments_per_conv[i] if sql_fragments_per_conv else None
                    prompt = get_best_user_answer_technical_prompt(
                        self.db, conversations[i], gens[a_idx], gens[b_idx], frags)
                    model_class = BestUserAnswerTechnicalResponse
                    extractor = get_best_user_answer_technical_result
                else:  # IRRELEVANT
                    prompt = get_best_user_answer_irrelevant_prompt(
                        self.db, conversations[i], gens[a_idx], gens[b_idx])
                    model_class = BestUserAnswerIrrelevantResponse
                    extractor = get_best_user_answer_irrelevant_result

                pairwise_prompts[i].append((a_idx, b_idx, prompt, model_class, extractor))

        # Flatten prompts into a single batch
        prompts: list[str] = []
        model_classes: list[type[BaseModel]] = []
        extractors: list[Callable] = []
        for i in range(len(answers)):
            for _, _, prompt, model_class, extractor in pairwise_prompts[i]:
                prompts.append(prompt)
                model_classes.append(model_class)
                extractors.append(extractor)

        responses_per_model: list[list[BaseModel]] = []
        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(prompts, model_classes)
            model.close()
            responses_per_model.append(responses)

        # Aggregate votes
        for model_responses in responses_per_model:
            response_idx = 0
            for i in range(len(answers)):
                comparisons = pairwise_prompts[i]
                for j in range(len(comparisons)):
                    response = model_responses[response_idx]
                    a_idx, b_idx, _, _, extractor = comparisons[j]
                    winner = extractor(response)
                    if winner == 0:
                        votes[i][a_idx] += 1
                    elif winner == 1:
                        votes[i][b_idx] += 1
                    response_idx += 1

        # Select best generation per conversation
        best_generations: list[str] = []
        tournament_tracking: list[TournamentVotes] = []
        for i, gens in enumerate(answers):
            best_idx = votes[i].index(max(votes[i]))
            best_generations.append(gens[best_idx])

            # Build pairwise results for tracking
            pairwise_for_tracking: list[dict] = []
            response_idx_offset = sum(len(pairwise_prompts[j]) for j in range(i))
            for pidx_inner, (a_idx, b_idx, _, _, extractor_fn) in enumerate(pairwise_prompts[i]):
                for midx, model_responses in enumerate(responses_per_model):
                    response = model_responses[response_idx_offset + pidx_inner]
                    winner = extractor_fn(response)
                    pairwise_for_tracking.append({
                        "model": self.models[midx].model_name,
                        "candidate_a": a_idx,
                        "candidate_b": b_idx,
                        "winner": winner,
                    })

            if candidate_model_indices is not None:
                original_model_idx = candidate_model_indices[i][best_idx]
                winning_model = self.models[original_model_idx].model_name
            else:
                winning_model = self.models[best_idx].model_name if best_idx < len(self.models) else f"model_{best_idx}"
            tallies = votes[i]
            max_votes = max(tallies)
            sorted_tallies = sorted(tallies, reverse=True)
            second_max = sorted_tallies[1] if len(sorted_tallies) > 1 else 0

            tournament_tracking.append(TournamentVotes(
                question_index=i,
                interaction_step=len(conversations[i].interactions) - 1,
                pairwise_results=pairwise_for_tracking,
                final_tallies=tallies,
                winning_answer_model=winning_model,
                margin=max_votes - second_max,
            ))

        return best_generations, tournament_tracking
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_best_user_answer.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add users/best_user_answer.py users/prompts/best_user_answer_prompt.py tests/test_best_user_answer.py
git commit -m "feat: switch tournament to combinations with randomized ordering, update TECHNICAL criteria"
```

---

### Task 6: Rewrite UserResponse orchestrator for two-stage flow

**Files:**
- Rewrite: `users/user_response.py`
- Create: `tests/test_user_response.py`

- [ ] **Step 1: Write failing tests for two-stage flow**

Create `tests/test_user_response.py`:

```python
"""Tests for two-stage UserResponse orchestrator."""
import pytest
from unittest.mock import MagicMock, patch
from dataset_dataclasses.benchmark import Conversation, Interaction, SystemResponse, RelevancyLabel, CategoryUse
from dataset_dataclasses.question import Question, QuestionUnanswerable, QuestionStyle, QuestionDifficulty
from users.user_response import UserResponse


def _make_category(is_answerable=False, is_solvable=True):
    cat = MagicMock()
    cat.is_answerable.return_value = is_answerable
    cat.is_solvable.return_value = is_solvable
    return cat


def _make_unsolvable_conv():
    cat = _make_category(is_answerable=False, is_solvable=False)
    q = QuestionUnanswerable(
        db_id="test_db", question="Q", evidence=None, sql=None,
        category=cat, question_style=QuestionStyle.FORMAL,
        question_difficulty=QuestionDifficulty.SIMPLE,
        hidden_knowledge="Missing data", is_solvable=False,
    )
    return Conversation(
        question=q,
        interactions=[Interaction(system_response=SystemResponse(system_question="What?"))],
        category_use=CategoryUse.GROUND_TRUTH,
    )


class TestShortCircuitUnsolvable:
    def test_unsolvable_gets_irrelevant_and_canned_refusal(self):
        db = MagicMock()
        model = MagicMock()
        model.model_name = "test-model"
        ur = UserResponse(db, [model])

        conv = _make_unsolvable_conv()
        ur.get_response([conv])

        assert conv.interactions[-1].relevance == RelevancyLabel.IRRELEVANT
        assert conv.interactions[-1].user_response == "That's not relevant to my question."
        # Model should NOT have been called
        model.init.assert_not_called()


def _make_solvable_conv(system_question="What do you mean by 'recent'?"):
    cat = _make_category(is_answerable=False, is_solvable=True)
    q = QuestionUnanswerable(
        db_id="test_db", question="Show me recent students", evidence=None,
        sql="SELECT name FROM students ORDER BY enrollment_date DESC LIMIT 10",
        category=cat, question_style=QuestionStyle.COLLOQUIAL,
        question_difficulty=QuestionDifficulty.SIMPLE,
        hidden_knowledge="Recent means enrolled in the last 6 months",
        is_solvable=True,
    )
    return Conversation(
        question=q,
        interactions=[Interaction(system_response=SystemResponse(system_question=system_question))],
        category_use=CategoryUse.GROUND_TRUTH,
    )


def _make_answerable_conv(system_question="How many results do you want?"):
    cat = _make_category(is_answerable=True, is_solvable=True)
    q = Question(
        db_id="test_db", question="List all students", evidence=None,
        sql="SELECT name FROM students ORDER BY name ASC LIMIT 20",
        category=cat, question_style=QuestionStyle.FORMAL,
        question_difficulty=QuestionDifficulty.SIMPLE,
    )
    return Conversation(
        question=q,
        interactions=[Interaction(system_response=SystemResponse(system_question=system_question))],
        category_use=CategoryUse.GROUND_TRUTH,
    )


def _mock_model(name, classify_response, answer_response):
    """Create a mock model that returns classify_response for Stage 1 and answer_response for Stage 2."""
    model = MagicMock()
    model.model_name = name
    # Track call count to alternate between Stage 1 and Stage 2
    call_count = [0]

    def mock_batch(prompts, model_classes):
        call_count[0] += 1
        if call_count[0] == 1:
            # Stage 1: classification
            return [classify_response] * len(prompts)
        else:
            # Stage 2: answer generation
            return [answer_response] * len(prompts)

    model.generate_batch_with_constraints = MagicMock(side_effect=mock_batch)
    return model


class TestTwoStageFlowSolvable:
    def test_solvable_relevant_majority(self):
        """Three models vote: 2 RELEVANT + 1 TECHNICAL -> RELEVANT wins."""
        from users.prompts.user_classify_prompt import UserClassifySolvable
        from users.prompts.user_answer_prompt import UserAnswerModel

        db = MagicMock()
        relevant_classify = UserClassifySolvable(relevancy="Relevant", node_ids=[])
        technical_classify = UserClassifySolvable(relevancy="Technical", node_ids=[1])
        answer = UserAnswerModel(answer="I mean the last 6 months")

        m1 = _mock_model("m1", relevant_classify, answer)
        m2 = _mock_model("m2", relevant_classify, answer)
        m3 = _mock_model("m3", technical_classify, answer)

        ur = UserResponse(db, [m1, m2, m3])
        conv = _make_solvable_conv()
        ur.get_response([conv])

        assert conv.interactions[-1].relevance == RelevancyLabel.RELEVANT
        assert conv.interactions[-1].user_response == "I mean the last 6 months"

    def test_solvable_technical_unions_node_ids(self):
        """Two models vote TECHNICAL with different nodes -> union taken."""
        from users.prompts.user_classify_prompt import UserClassifySolvable
        from users.prompts.user_answer_prompt import UserAnswerModel

        db = MagicMock()
        tech1 = UserClassifySolvable(relevancy="Technical", node_ids=[1, 2])
        tech2 = UserClassifySolvable(relevancy="Technical", node_ids=[2, 3])
        answer = UserAnswerModel(answer="Sort by newest first, top 10")

        m1 = _mock_model("m1", tech1, answer)
        m2 = _mock_model("m2", tech2, answer)

        ur = UserResponse(db, [m1, m2])
        conv = _make_solvable_conv()
        ur.get_response([conv])

        assert conv.interactions[-1].relevance == RelevancyLabel.TECHNICAL
        assert conv.interactions[-1].user_response == "Sort by newest first, top 10"

    def test_tie_resolves_to_irrelevant(self):
        """1 RELEVANT + 1 TECHNICAL + 1 IRRELEVANT -> tie -> IRRELEVANT."""
        from users.prompts.user_classify_prompt import UserClassifySolvable
        from users.prompts.user_answer_prompt import UserAnswerModel

        db = MagicMock()
        r = UserClassifySolvable(relevancy="Relevant", node_ids=[])
        t = UserClassifySolvable(relevancy="Technical", node_ids=[1])
        ir = UserClassifySolvable(relevancy="Irrelevant", node_ids=[])
        answer = UserAnswerModel(answer="Not relevant")

        m1 = _mock_model("m1", r, answer)
        m2 = _mock_model("m2", t, answer)
        m3 = _mock_model("m3", ir, answer)

        ur = UserResponse(db, [m1, m2, m3])
        conv = _make_solvable_conv()
        ur.get_response([conv])

        assert conv.interactions[-1].relevance == RelevancyLabel.IRRELEVANT


class TestTwoStageFlowAnswerable:
    def test_answerable_technical(self):
        """Answerable question classified as TECHNICAL."""
        from users.prompts.user_classify_prompt import UserClassifyAnswerable
        from users.prompts.user_answer_prompt import UserAnswerModel

        db = MagicMock()
        classify = UserClassifyAnswerable(relevancy="Technical", node_ids=[1])
        answer = UserAnswerModel(answer="I'd like 20 results please")

        m1 = _mock_model("m1", classify, answer)
        ur = UserResponse(db, [m1])
        conv = _make_answerable_conv()
        ur.get_response([conv])

        assert conv.interactions[-1].relevance == RelevancyLabel.TECHNICAL
        assert conv.interactions[-1].user_response == "I'd like 20 results please"


class TestSingleModelSkipsTournament:
    def test_single_model_no_tournament(self):
        """With 1 model, tournament is skipped and answer is used directly."""
        from users.prompts.user_classify_prompt import UserClassifySolvable
        from users.prompts.user_answer_prompt import UserAnswerModel

        db = MagicMock()
        classify = UserClassifySolvable(relevancy="Relevant", node_ids=[])
        answer = UserAnswerModel(answer="I mean recent as in last 6 months")

        m1 = _mock_model("m1", classify, answer)
        ur = UserResponse(db, [m1])
        conv = _make_solvable_conv()
        relevancy_tracking, tournament_tracking = ur.get_response([conv])

        assert conv.interactions[-1].user_response == "I mean recent as in last 6 months"
        assert tournament_tracking == []  # No tournament with 1 model
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_user_response.py -v`
Expected: FAIL (old imports in user_response.py)

- [ ] **Step 3: Rewrite user_response.py**

Rewrite `users/user_response.py` with the two-stage flow:

```python
"""Two-stage user response: Stage 1 classifies and locates AST nodes,
Stage 2 generates natural-language-only answers grounded in matched source.

Replaces the previous single-stage pipeline. Following BIRD-Interact's
(Li et al., 2025) function-driven approach adapted for ABISS.
"""

from typing import Callable
from db_datasets.db_dataset import DBDataset
from models.model import Model
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.question import QuestionUnanswerable
from dataset_dataclasses.council_tracking import RelevancyVotes, TournamentVotes
from users.best_user_answer import BestUserAnswer
from users.sql_ast import parse_sql_to_nodes, SQLNode
from users.prompts.user_classify_prompt import (
    UserClassifySolvable,
    UserClassifyAnswerable,
    get_user_classify_prompt_solvable,
    get_user_classify_prompt_answerable,
    get_classify_solvable_result,
    get_classify_answerable_result,
)
from users.prompts.user_answer_prompt import (
    UserAnswerModel,
    get_user_answer_prompt_relevant,
    get_user_answer_prompt_technical,
    get_user_answer_prompt_irrelevant,
)
from pydantic import BaseModel


class UserResponse:
    """Two-stage user response pipeline."""

    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db = db
        self.models = models
        self.best_user_answer = BestUserAnswer(db, models)

    def get_response(self, conversations: list[Conversation]) -> tuple[list[RelevancyVotes], list[TournamentVotes]]:
        # ---- short-circuit unsolvable questions ----
        to_process: list[int] = []

        for i, conv in enumerate(conversations):
            is_solvable = (
                conv.question.category.is_solvable()
                if isinstance(conv.question, QuestionUnanswerable)
                else True
            )

            if not is_solvable:
                conv.interactions[-1].relevance = RelevancyLabel.IRRELEVANT
                conv.interactions[-1].user_response = "That's not relevant to my question."
                continue

            to_process.append(i)

        if not to_process:
            return [], []

        # ---- pre-parse GT SQL into AST nodes ----
        nodes_per_conv: list[list[SQLNode]] = []
        for idx in to_process:
            sql = conversations[idx].question.sql
            nodes_per_conv.append(parse_sql_to_nodes(sql))

        # ---- STAGE 1: Classification + Source Identification ----
        stage1_prompts: list[str] = []
        stage1_model_classes: list[type[BaseModel]] = []
        stage1_extractors: list[Callable] = []

        for pidx, conv_idx in enumerate(to_process):
            conv = conversations[conv_idx]
            nodes = nodes_per_conv[pidx]
            is_answerable = conv.question.category.is_answerable()

            if is_answerable:
                stage1_prompts.append(get_user_classify_prompt_answerable(conv, nodes))
                stage1_model_classes.append(UserClassifyAnswerable)
                stage1_extractors.append(get_classify_answerable_result)
            else:
                stage1_prompts.append(get_user_classify_prompt_solvable(conv, nodes))
                stage1_model_classes.append(UserClassifySolvable)
                stage1_extractors.append(get_classify_solvable_result)

        # Collect per-model classifications
        relevancy_votes: list[dict[RelevancyLabel, int]] = [
            {RelevancyLabel.RELEVANT: 0, RelevancyLabel.TECHNICAL: 0, RelevancyLabel.IRRELEVANT: 0}
            for _ in stage1_prompts
        ]
        per_model_node_ids: list[list[list[int]]] = [[] for _ in stage1_prompts]
        per_model_labels: list[list[RelevancyLabel]] = [[] for _ in stage1_prompts]

        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(stage1_prompts, stage1_model_classes)
            model.close()

            for pidx, response in enumerate(responses):
                label, node_ids = stage1_extractors[pidx](response)
                relevancy_votes[pidx][label] += 1
                per_model_labels[pidx].append(label)
                per_model_node_ids[pidx].append(node_ids)

        # ---- Majority vote on labels ----
        final_labels: list[RelevancyLabel] = []
        for pidx in range(len(stage1_prompts)):
            votes = relevancy_votes[pidx]
            sorted_labels = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
            top_label, top_count = sorted_labels[0]
            _, second_count = sorted_labels[1]
            if top_count == second_count:
                final_labels.append(RelevancyLabel.IRRELEVANT)
            else:
                final_labels.append(top_label)

        # ---- Resolve sources based on winning label ----
        # For TECHNICAL: union of node_ids from models that voted TECHNICAL
        resolved_fragments: list[list[str]] = []
        for pidx in range(len(stage1_prompts)):
            if final_labels[pidx] == RelevancyLabel.TECHNICAL:
                union_ids: set[int] = set()
                for midx in range(len(self.models)):
                    if per_model_labels[pidx][midx] == RelevancyLabel.TECHNICAL:
                        union_ids.update(per_model_node_ids[pidx][midx])
                # Map node_ids to SQL fragments
                nodes = nodes_per_conv[pidx]
                node_map = {n.node_id: n.sql_fragment for n in nodes}
                fragments = [node_map[nid] for nid in sorted(union_ids) if nid in node_map]
                resolved_fragments.append(fragments)
            else:
                resolved_fragments.append([])

        # ---- Build relevancy tracking ----
        relevancy_tracking: list[RelevancyVotes] = []
        for pidx, conv_idx in enumerate(to_process):
            model_labels = [
                (self.models[midx].model_name, per_model_labels[pidx][midx].value)
                for midx in range(len(self.models))
            ]
            all_same = len(set(label for _, label in model_labels)) == 1
            relevancy_tracking.append(RelevancyVotes(
                question_index=conv_idx,
                interaction_step=len(conversations[conv_idx].interactions) - 1,
                per_model_labels=model_labels,
                winning_label=final_labels[pidx].value,
                unanimous=all_same,
            ))

        # ---- Assign labels ----
        for pidx, conv_idx in enumerate(to_process):
            conversations[conv_idx].interactions[-1].relevance = final_labels[pidx]

        # ---- STAGE 2: Response Generation ----
        stage2_prompts: list[str] = []
        for pidx, conv_idx in enumerate(to_process):
            conv = conversations[conv_idx]
            label = final_labels[pidx]
            if label == RelevancyLabel.RELEVANT:
                stage2_prompts.append(get_user_answer_prompt_relevant(conv))
            elif label == RelevancyLabel.TECHNICAL:
                stage2_prompts.append(get_user_answer_prompt_technical(conv, resolved_fragments[pidx]))
            else:
                stage2_prompts.append(get_user_answer_prompt_irrelevant(conv))

        stage2_model_classes = [UserAnswerModel] * len(stage2_prompts)

        per_model_answers: list[list[str]] = [[] for _ in stage2_prompts]
        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(stage2_prompts, stage2_model_classes)
            model.close()

            for pidx, response in enumerate(responses):
                validated = UserAnswerModel.model_validate(response)
                per_model_answers[pidx].append(validated.answer.strip())

        # ---- Select best answer via tournament ----
        best_answers: list[str] = []
        needs_tournament_indices: list[int] = []
        needs_tournament_convs: list[Conversation] = []
        needs_tournament_answers: list[list[str]] = []
        needs_tournament_fragments: list[list[str]] = []

        for pidx in range(len(stage2_prompts)):
            answers = per_model_answers[pidx]
            if len(answers) == 1:
                best_answers.append(answers[0])
            else:
                best_answers.append("")  # placeholder
                needs_tournament_indices.append(pidx)
                needs_tournament_convs.append(conversations[to_process[pidx]])
                needs_tournament_answers.append(answers)
                needs_tournament_fragments.append(resolved_fragments[pidx])

        tournament_tracking: list[TournamentVotes] = []
        if needs_tournament_convs:
            tournament_results, tournament_tracking = self.best_user_answer.select_best_user_answers(
                needs_tournament_convs, needs_tournament_answers,
                sql_fragments_per_conv=needs_tournament_fragments,
            )
            for i, pidx in enumerate(needs_tournament_indices):
                best_answers[pidx] = tournament_results[i]

        for pidx, conv_idx in enumerate(to_process):
            conversations[conv_idx].interactions[-1].user_response = best_answers[pidx]

        return relevancy_tracking, tournament_tracking
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_user_response.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add users/user_response.py tests/test_user_response.py
git commit -m "feat: rewrite UserResponse with two-stage classification + generation flow"
```

---

### Task 7: Delete old files and clean up imports

**Files:**
- Delete: `users/sql_preferences.py`
- Delete: `users/prompts/user_response_prompt.py`
- Verify: no remaining imports of deleted modules

- [ ] **Step 1: Check for remaining references to deleted modules**

Run: `grep -r "sql_preferences" --include="*.py" .`
Run: `grep -r "user_response_prompt" --include="*.py" .`

Expected: No references outside the files being deleted and the old `best_user_answer_prompt.py` (which was already updated in Task 5).

- [ ] **Step 2: Delete old files**

```bash
rm users/sql_preferences.py
rm users/prompts/user_response_prompt.py
```

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Verify the full import chain works**

Run: `python -c "from users.user import User; print('OK')"`
Expected: Prints "OK" with no import errors.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove old single-stage user response files (sql_preferences, user_response_prompt)"
```

---

### Task 8: End-to-end smoke test

**Files:**
- No new files; verify existing integration.

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 2: Verify import chain from do_interaction.py**

Run: `python -c "from do_interaction import main; print('Import OK')"`
Expected: Prints "Import OK" (may require vllm/torch, but should not crash on the new user simulator imports).

- [ ] **Step 3: Commit any final fixes**

If any fixes were needed, commit them:
```bash
git add -A
git commit -m "fix: resolve integration issues from two-stage user simulator migration"
```
