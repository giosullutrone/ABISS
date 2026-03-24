# Two-Stage User Simulator with AST-Based Retrieval

**Date:** 2026-03-24
**Status:** Approved
**Scope:** Replace the single-stage user response pipeline with a two-stage classifier/locator + response generator, following BIRD-Interact's function-driven approach adapted for ABISS.

## Dependencies

- `sqlglot` must be added to `requirements.txt`. Required for SQLite dialect AST parsing.

## Motivation

The current user simulator has three problems:

1. **Incomplete SQL coverage.** Only ORDER BY, LIMIT, and DISTINCT are extracted from GT SQL. When the system asks about GROUP BY, HAVING, or aggregation functions, the user agent defaults to uncertainty, potentially steering the system toward wrong SQL.
2. **Raw SQL leakage.** The extracted preferences include raw SQL fragments (e.g., `T1.enrollment_date DESC`), which a real non-technical user would never know.
3. **Coupled classification and response.** Classification and answer generation happen in a single LLM call. Models that disagree on classification produce answers under different intents, requiring post-hoc filtering that discards valid responses.

The new design addresses all three by adopting BIRD-Interact's (Li et al., 2025) AST-based retrieval mechanism: parse GT SQL into addressable clause-level nodes, let the classifier identify which nodes are relevant to a clarification question, then generate a natural-language-only response grounded in those nodes.

## Architecture Overview

```
GT SQL --> [sqlglot AST parse] --> numbered node list (once per question)

Stage 1 (Encoder/Classifier):
  Clarification Q + hidden_knowledge + AST nodes (raw SQL) --> [LLM] --> (label, node_ids[])
  Council majority-votes on label. Union of node_ids for TECHNICAL.

Stage 2 (Decoder/Generator):
  Clarification Q + matched source --> [LLM] --> answer (NL only)
  Source = hidden_knowledge (RELEVANT) | SQL fragments (TECHNICAL) | nothing (IRRELEVANT)
  Council generates answers --> tournament selects best.
```

## Component 1: SQL AST Parser

**New file:** `users/sql_ast.py` (replaces `users/sql_preferences.py`)

### Responsibilities

- Parse GT SQL into an AST using sqlglot (SQLite dialect).
- Extract clause-level nodes, each with: `node_id` (int), `clause_type` (str), `sql_fragment` (str).
- Provide `format_nodes_for_prompt()` to render nodes as a numbered list with raw SQL.

### Clause types extracted

| Clause type | What it captures |
|---|---|
| SELECT | Individual select expressions (columns, aggregations) |
| WHERE | Filtering conditions (excluding join conditions) |
| GROUP_BY | Grouping columns |
| HAVING | Group-level filter conditions |
| ORDER_BY | Sort specifications |
| LIMIT | Result count restrictions |
| DISTINCT | Uniqueness flag |
| JOIN | Join clauses (table + condition) |

### Prompt format

```
[1] SELECT: COUNT(DISTINCT T1.student_id)
[2] WHERE: T1.grade_level = 12
[3] GROUP BY: T1.department
[4] ORDER BY: COUNT(*) DESC
[5] LIMIT: 10
```

### Edge cases

- `sql is None` (unanswerable questions): returns empty node list. These are short-circuited before reaching the two-stage pipeline.
- Unparseable SQL: fall back to empty node list with a warning. The user agent will only have `hidden_knowledge` to work with.

## Component 2: Stage 1 - Classifier/Locator

**New file:** `users/prompts/user_classify_prompt.py`

### Pydantic models

```python
class UserClassifySolvable(BaseModel):
    relevancy: Literal["Relevant", "Technical", "Irrelevant"]
    node_ids: list[int]  # 1+ nodes for TECHNICAL, empty otherwise

class UserClassifyAnswerable(BaseModel):
    relevancy: Literal["Technical", "Irrelevant"]
    node_ids: list[int]
```

Multi-node selection is allowed. The LLM can reference multiple AST nodes when a clarification question spans several SQL clauses.

The relationship between `relevancy` and `node_ids` is enforced by the orchestrator during Stage 1 aggregation, not by the Pydantic model. If a TECHNICAL vote has empty `node_ids`, the orchestrator treats it as a valid TECHNICAL vote with no node contribution to the union.

### Prompt structure

The Stage 1 prompt receives:

- Original question + evidence
- Conversation history (completed turns only)
- Hidden knowledge (included only when `category.is_solvable() == True` and `question` is a `QuestionUnanswerable` instance; omitted for answerable questions)
- Full AST node list with raw SQL fragments
- The current clarification question
- Classification definitions (RELEVANT / TECHNICAL / IRRELEVANT)
- Instruction: "If TECHNICAL, list ALL node IDs the question relates to. If RELEVANT or IRRELEVANT, leave node_ids empty."

### Council aggregation

- **Label:** Majority vote across all council models. Ties resolve to IRRELEVANT (unchanged from current behavior).
- **Node IDs (TECHNICAL only):** Union of all node IDs referenced by models that voted TECHNICAL. This gives broad coverage rather than filtering to consensus.

## Component 3: Stage 2 - Response Generator

**New file:** `users/prompts/user_answer_prompt.py`

### Pydantic model

```python
class UserAnswerModel(BaseModel):
    answer: str
```

### Prompt variants by label

**RELEVANT:**
- Source: `hidden_knowledge`
- Instruction: Use the hidden knowledge to disambiguate. Be direct about which interpretation you mean.

**TECHNICAL:**
- Source: The matched SQL fragment(s) from Stage 1 node resolution, provided verbatim in the prompt as source material. The NL-only constraint applies to the generated answer, not to the prompt input. The model is expected to paraphrase the SQL intent into natural language.
- Instruction: Answer the question based on the provided source information. Express the intent in natural language using the conversation's style (e.g., `ORDER BY COUNT(*) DESC` becomes "I'd like the most popular ones first" in colloquial style, or "Sort by frequency in descending order" in formal style). The SQL fragment is reference material for what the user wants; the answer must sound like a real non-technical user speaking in their own voice. If no source fragments are available (empty node list), express genuine uncertainty.

**IRRELEVANT:**
- Source: None
- Instruction: Politely but firmly refuse to answer.

### Key constraint (all variants)

> "You are a non-technical user. Respond in natural language ONLY. NEVER use SQL syntax, column names, table aliases, or query structure. Express the intent behind the information, not its SQL representation."

### Style matching

All variants include the question's style description and instruction to match the original question's register and vocabulary (unchanged from current behavior).

## Component 4: Modified UserResponse Orchestrator

**Modified file:** `users/user_response.py`

### New flow

1. **Short-circuit unsolvable** (unchanged): Set IRRELEVANT + canned refusal.
2. **Pre-parse GT SQL:** Call `parse_sql_to_nodes()` once per conversation in the current batch. Since `question.sql` is immutable, re-parsing across interaction steps is acceptable and avoids caching complexity. If `question.sql` is None for a non-short-circuited question, the empty node list means TECHNICAL classifications will produce answers with no source material; the Stage 2 TECHNICAL prompt falls back to an uncertainty response.
3. **Stage 1:** Build a single mixed batch of classification prompts (mixing `UserClassifySolvable` and `UserClassifyAnswerable` models, same pattern as current code). For each council model, run `generate_batch_with_constraints`. Collect `(label, node_ids)` per model per conversation.
4. **Aggregate Stage 1:** Majority-vote labels. Union node_ids for TECHNICAL winners. If a TECHNICAL vote has empty `node_ids`, it counts as a valid TECHNICAL vote with no node contribution to the union.
5. **Resolve sources:** For each conversation, build the source material based on winning label:
   - RELEVANT: `hidden_knowledge`
   - TECHNICAL: SQL fragments from the union of matched node IDs
   - IRRELEVANT: None
6. **Stage 2:** Build answer prompts with resolved sources. For each council model, run `generate_batch_with_constraints`. Collect answers.
7. **Tournament:** If 2+ answers, run pairwise tournament. If 1, use directly.

### Key change from current design

All Stage 2 answers are generated under the same winning label and source. There is no longer a need to filter for "label-consistent" answers, since the classification is resolved before response generation.

## Component 5: Modified Tournament

**Modified file:** `users/best_user_answer.py`

### Combinations instead of permutations

Replace full permutation pairs (A-vs-B and B-vs-A) with combinations only (A-vs-B once). This halves the number of tournament comparisons.

Change: Replace nested `for j / for k where j != k` with `itertools.combinations(range(len(candidates)), 2)`.

To mitigate positional bias (LLMs tend to favor the first option), randomize which candidate appears as "A" vs "B" within each combination.

### Modified file: `users/prompts/best_user_answer_prompt.py`

**TECHNICAL evaluation criteria change.** Instead of checking against ORDER BY/LIMIT/DISTINCT preferences, the evaluator receives the matched SQL fragment(s) and checks:

- Does the answer accurately convey the intent behind the source information?
- Does it avoid revealing SQL syntax, column names, or table aliases?
- Does it avoid fabricating preferences not grounded in the source?

RELEVANT and IRRELEVANT tournament prompts remain essentially unchanged.

## Files Summary

| File | Action | Description |
|---|---|---|
| `users/sql_ast.py` | Create | AST parser using sqlglot |
| `users/prompts/user_classify_prompt.py` | Create | Stage 1 classification prompts |
| `users/prompts/user_answer_prompt.py` | Create | Stage 2 response generation prompts |
| `users/user_response.py` | Rewrite | Two-stage orchestration flow |
| `users/best_user_answer.py` | Modify | Combinations not permutations |
| `users/prompts/best_user_answer_prompt.py` | Modify | TECHNICAL criteria use matched fragments |
| `users/sql_preferences.py` | Delete | Replaced by sql_ast.py |
| `users/prompts/user_response_prompt.py` | Delete | Replaced by Stage 1 + Stage 2 prompts |

## Unchanged Files

- `benchmarks/benchmark.py` (calls `user.get_response()`, interface unchanged)
- `users/user.py` (delegates to `UserResponse`, interface unchanged)
- `dataset_dataclasses/benchmark.py` (RelevancyLabel, Interaction, Conversation unchanged)
- `dataset_dataclasses/question.py` (Question, QuestionUnanswerable unchanged)
- `dataset_dataclasses/council_tracking.py` (node IDs are ephemeral and not persisted in tracking; `RelevancyVotes` unchanged)
- All evaluators (recognition, classification, generation, feedback)
- All system agent code (agents/)

## Cost Impact

Two LLM calls per model per interaction instead of one. With a 3-model council:
- Current: 3 calls (classify + respond) + tournament calls
- New: 6 calls (3 classify + 3 respond) + tournament calls (halved by combinations)

Stage 1 outputs are small structured objects (label + node list). The added token cost is modest. The tournament cost reduction partially offsets the Stage 1 addition.

## References

- Li et al. (2025). "BIRD-Interact: Re-imagining Text-to-SQL Evaluation for Large Language Models via Lens of Dynamic Interactions." arXiv:2510.05318.
