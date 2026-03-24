# Split SQL Generation Design

## Problem

The current pipeline generates question text and SQL in a single LLM call. This couples two distinct tasks, and the SQL quality suffers because the model splits its attention between crafting a good question and writing correct SQL. Our annotation results show SQL correctness at ~70-77% across datasets.

## Solution

Split question generation into two phases:

1. **Phase 1 (Question Generation):** Generate question text, hidden knowledge, evidence, and feedback (no SQL).
2. **Phase 2 (SQL Generation):** For each question that needs SQL (answerable + ambiguous), generate SQL using a DIN-SQL-inspired chain-of-thought prompt.

Unanswerable questions are unchanged (they never have SQL).

## Design Decisions

- **Same model** for both phases (no separate model configuration).
- **Single-pass DIN-SQL** (schema linking + sub-question decomposition + SQL in one chain-of-thought call, not three separate calls).
- **No DIN-SQL classification step** since difficulty is already assigned from phase 1. SIMPLE/MODERATE map to easy/medium prompts; COMPLEX/HIGHLY_COMPLEX both map to hard (with sub-question decomposition).
- **Phase 2 runs inside `generate_for_model`** (not as a validation step), keeping SQL generation as part of the generation phase.
- **Null SQL removal:** Questions where the SQL generation returns None are dropped from the dataset immediately, before reaching validators. This guarantees that validators with `assert question.sql is not None` (SQLExecutability, GTSatisfaction, AmbiguityVerification) never receive sql-less solvable questions.
- **Phase 2 uses `n_samples=1`** (one SQL per question) since the question text is already fixed. Phase 1 uses the configured `n_samples` as before.
- **Unified SQL generation prompt:** The new DIN-SQL-inspired prompt replaces the existing `db_datasets/sql_generation_prompts.py`. Both the generator (Phase 2) and validators (`generate_sqls_unsafe`, `generate_sqls_without_evidence_unsafe`) use the same prompt.
- **Difficulty re-assignment after Phase 2:** Instead of validating difficulty conformance (and rejecting mismatches), re-assign `question.question_difficulty` based on the actual SQL produced using `classify_sql_difficulty()`. The `DifficultyConformance` validator is removed from the validation pipeline.

## Phase 1: Question-Only Generation

### Category Output Model Changes

Remove all SQL fields from the output models of answerable and ambiguous categories. The `get_question()` methods create Question objects with `sql=None`.

| Category | Output Class | Fields Removed |
|----------|-------------|----------------|
| AnswerableWithoutEvidence | AnswerableOutput | `sql` |
| AnswerableWithEvidence | AnswerableWithEvidenceOutput | `sql` |
| LexicalVagueness | LexicalVaguenessOutput | `sql_first_interpretation`, `sql_second_interpretation` |
| SemanticMappingEntityAmbiguity | SemanticMappingEntityAmbiguityOutput | `sql_first_entity`, `sql_second_entity` |
| SemanticMappingLexicalOverlap | SemanticMappingLexicalOverlapOutput | `sql_first_mapping`, `sql_second_mapping` |
| StructureAmbiguityAttachment | StructureAmbiguityAttachmentOutput | `sql_last_only`, `sql_all_elements` |
| StructureAmbiguityScope | StructureAmbiguityScopeOutput | `sql_collective`, `sql_distributive` |
| ConflictingKnowledge | ConflictingKnowledgeOutput | `sql_first_evidence`, `sql_second_evidence` (evidence fields kept) |
| MissingUserKnowledge | MissingUserKnowledgeOutput | `sql_with_user_knowledge` |

Unanswerable categories (MissingExternalKnowledge, MissingSchemaEntities, MissingSchemaRelationships, ImproperQuestion) are unchanged.

### Generation Prompt Changes (`generator_prompt.py`)

For answerable and solvable categories, remove all SQL-related content:
- **Task Overview** (lines 26-42): Remove mentions of "converted to SQL queries" and "SQL query" from all three branches.
- **SQL Difficulty Requirements** (lines 67-74): Keep as-is. Difficulty is a feature of the dataset and guides question complexity. The prompt still tells the model what difficulty level to target. However, after Phase 2, the actual difficulty is re-assigned based on the generated SQL (see "Difficulty Re-assignment" section).
- **Quality Guidelines** (lines 80-104): Remove all SQL-specific bullets: "SQL query is valid and executable", "SQL queries structurally different", "SQL is semantically correct", "SQL is self-contained", "SQL complexity matches difficulty", "Use explicit aliases".
- **Generation Process** (lines 106-123): Remove `**sql**` and `**sql_with_user_knowledge**` references; simplify to only iterate on question and hidden_knowledge fields.
- **Output Format** (line 78): Uses the updated SQL-free output models via `model_field_descriptions(output)` (no change needed here, it auto-reflects the model).

### `get_question()` Method Changes

Each affected category's `get_question()` sets `sql=None`:

- **Answerable categories:** Create `Question(..., sql=None)`
- **Ambiguous categories with two interpretations:** The list comprehension still splits into two QuestionUnanswerable objects, but uses `sql=None` for each. Hidden knowledge is still assigned per interpretation.
- **MissingUserKnowledge:** Creates `QuestionUnanswerable(..., sql=None)`

## Phase 2: DIN-SQL-Inspired SQL Generation

### New File: `generators/prompts/sql_generation_prompt.py`

Contains:
- `SQLGenerationOutput(BaseModel)` with a single field: `sql: str`
- `get_sql_generation_prompt(db: DBDataset, question: Question | QuestionUnanswerable) -> str`

The function reads `question.question`, `question.evidence`, `question.question_difficulty`, `question.db_id`, and uses `isinstance(question, QuestionUnanswerable)` to check for `hidden_knowledge`.

### Prompt Structure

The prompt is a single chain-of-thought that adapts based on difficulty:

```
1. ROLE: Expert SQL developer for text-to-SQL systems.

2. DATABASE SCHEMA: (using existing db.get_schema_prompt(db_id, rows=5))

3. QUESTION: {question.question}
   EVIDENCE: {question.evidence}  (if any)
   HIDDEN KNOWLEDGE: {question.hidden_knowledge}  (if ambiguous)

4. CHAIN-OF-THOUGHT INSTRUCTIONS:

   Step 1 - Schema Linking:
   Identify relevant tables, columns, and foreign key relationships
   for this question. List them explicitly.

   Step 2 - Sub-question Decomposition (COMPLEX/HIGHLY_COMPLEX only):
   Break the question into simpler sub-questions. For each sub-question,
   identify the SQL operation needed.

   Step 3 - SQL Generation:
   Write the final SQL query.
   - Use double quotes for identifiers with spaces or special characters.
   - Use explicit aliases for aggregated columns.
   - The query must be valid and executable against the provided schema.
   - For ambiguous questions: the SQL must faithfully represent the
     interpretation described in the hidden knowledge.

5. OUTPUT: JSON with sql field.
```

**Difficulty mapping in the prompt:**
- SIMPLE: "Write a simple query. No JOINs or nested queries should be needed."
- MODERATE: "The query may require JOINs across multiple tables."
- COMPLEX / HIGHLY_COMPLEX: "The query requires nested subqueries, set operations (UNION/INTERSECT/EXCEPT), or complex aggregations. Decompose the question into sub-questions first."

### Integration in `generate_for_model`

Updated flow:

```python
def generate_for_model(self, model, db_ids, categories, styles, difficulties):
    # --- Phase 1: Question-only generation (same as before but no SQL) ---
    prompts, constraints, metadata = build_question_prompts(...)  # as before
    model.init()
    responses = model.generate_batch_with_constraints_unsafe(prompts, constraints)
    # responses has len(prompts) * n_samples entries

    # Convert to Question objects (sql=None for answerable/ambiguous)
    # Uses existing n_samples indexing: metadata[idx // self.n_samples]
    questions = []
    for idx, response in enumerate(responses):
        if response is not None:
            category, db_id, style, difficulty = metadata[idx // self.n_samples]
            questions.extend(category.get_question(db_id, response, style, difficulty))

    # --- Phase 2: SQL generation for answerable + ambiguous ---
    # Phase 2 generates exactly 1 SQL per question (no n_samples multiplier).
    # The model must be called with n=1 for this batch.
    sql_questions = [q for q in questions if q.category.is_solvable()]
    sql_prompts = [get_sql_generation_prompt(self.db, q) for q in sql_questions]
    sql_constraints = [SQLGenerationOutput] * len(sql_prompts)

    sql_responses = model.generate_batch_with_constraints_unsafe(
        sql_prompts, sql_constraints, n_override=1
    )
    model.close()
    # sql_responses has exactly len(sql_prompts) entries (1:1)

    # Assign SQL and filter out failures
    questions_with_sql = []
    sql_idx = 0
    for q in questions:
        if q.category.is_solvable():
            resp = sql_responses[sql_idx]
            sql_idx += 1
            if resp is not None:
                q.sql = resp.sql
                questions_with_sql.append(q)
            # else: drop this question (SQL generation failed)
        else:
            questions_with_sql.append(q)  # unanswerable, keep as-is

    return questions_with_sql
```

Note: `n_override=1` may require a small addition to `generate_batch_with_constraints_unsafe` to accept an optional sample count override, or the model's `n` parameter can be temporarily set to 1 for the Phase 2 call.

## Difficulty Re-assignment

After Phase 2 assigns SQL to questions, re-assign `question.question_difficulty` based on the actual SQL complexity using the existing `classify_sql_difficulty()` function from `validators/difficulty_conformance.py`. This replaces the `DifficultyConformance` validator in the validation pipeline.

```python
# After Phase 2, before returning questions
from validators.difficulty_conformance import classify_sql_difficulty

for q in questions_with_sql:
    if q.sql is not None:
        q.question_difficulty = classify_sql_difficulty(q.sql)
```

This runs inside `generate_for_model`, after SQL assignment and null filtering. The `DifficultyConformance` validator is removed from the validation pipeline in `generator.py`.

## Unified SQL Generation Prompt

The new DIN-SQL-inspired prompt in `generators/prompts/sql_generation_prompt.py` replaces the existing prompt in `db_datasets/sql_generation_prompts.py`. The validators that use SQL generation (`ambiguity_verification.py`, `unsolvability_verification.py`) call `db.generate_sqls_unsafe()`, which uses `db_datasets/sql_generation_prompts.py`. This file is updated to use the new DIN-SQL-inspired prompt instead.

Changes to `db_datasets/sql_generation_prompts.py`:
- Replace the existing `get_sql_generation_prompt` with the new DIN-SQL-inspired version
- Keep `SQLGenerationResponse` and `get_sql_result` as-is (they are the output model used by validators)
- `generate_comment_prompt` is removed (superseded by the new prompt structure)

Changes to `db_datasets/db_dataset.py`:
- `generate_sqls_unsafe` and `generate_sqls_without_evidence_unsafe` continue to call the same function name, but get the improved prompt automatically

The Phase 2 prompt in `generators/prompts/sql_generation_prompt.py` imports and reuses the prompt function from `db_datasets/sql_generation_prompts.py` rather than duplicating it. `SQLGenerationOutput` in the generator is just a re-export or alias of `SQLGenerationResponse`.

## Files Changed

| File | Change |
|------|--------|
| `categories/answerable.py` | Remove `sql` field from output |
| `categories/answerable_with_evidence.py` | Remove `sql` field from output |
| `categories/lexical_vagueness.py` | Remove `sql_*` fields, update `get_question()` |
| `categories/semantic_mapping_entity_ambiguity.py` | Remove `sql_*` fields, update `get_question()` |
| `categories/semantic_mapping_lexical_overlap.py` | Remove `sql_*` fields, update `get_question()` |
| `categories/structure_ambiguity_attachment.py` | Remove `sql_*` fields, update `get_question()` |
| `categories/structure_ambiguity_scope.py` | Remove `sql_*` fields, update `get_question()` |
| `categories/conflicting_knowledge.py` | Remove `sql_*` fields, update `get_question()` |
| `categories/missing_user_knowledge.py` | Remove `sql_*` field, update `get_question()` |
| `generators/prompts/generator_prompt.py` | Remove SQL instructions for answerable/solvable (keep difficulty) |
| `generators/generator.py` | Two-phase flow in `generate_for_model`, difficulty re-assignment, remove DifficultyConformance from validation |
| `db_datasets/sql_generation_prompts.py` | Replace existing prompt with DIN-SQL-inspired version |

## Files NOT Changed

- `generators/chain.py`
- `validators/ambiguity_verification.py` (uses `generate_sqls_unsafe` which auto-gets new prompt)
- `validators/unsolvability_verification.py` (same)
- `db_datasets/db_dataset.py` (calls unchanged function name)
- `dataset_dataclasses/`
- Unanswerable category files (missing_external_knowledge, missing_schema_entities, missing_schema_relationships, improper_question)

## Trade-offs

- **More API calls:** Each answerable question requires 2 calls instead of 1. Ambiguous categories that split into 2 interpretations require 3 calls (1 question generation + 2 SQL). MissingUserKnowledge (single interpretation) requires 2 calls.
- **Better SQL quality:** The SQL generation prompt is purpose-built with chain-of-thought reasoning (schema linking, sub-question decomposition), which should improve correctness.
- **Cleaner separation:** Question quality and SQL quality can be evaluated and iterated independently.
- **Difficulty is always accurate:** Re-assigning difficulty from actual SQL eliminates mismatches and removes the DifficultyConformance validation step (which previously rejected questions).
