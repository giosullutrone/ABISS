# CLI Annotation Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a CLI script that presents sampled questions with full context and collects quality annotations via stdin, mirroring the Streamlit annotation tool.

**Architecture:** Single Python script (`annotate_questions_cli.py`) that reuses existing project modules (`annotate_questions.load_questions`, `annotate_questions.sample_questions`, `DBDataset`, `STYLE_DESCRIPTIONS`, category classes). Sequential question-by-question flow with autosave.

**Tech Stack:** Python 3.10+, SQLite (via existing `DBDataset`), existing project modules.

---

### Task 1: Create the CLI annotation script

**Files:**
- Create: `annotate_questions_cli.py`

**Step 1: Implement the script**

The script should:
1. Reuse `load_questions`, `sample_questions`, `execute_sql_query`, `get_database_schema_with_examples`, `_get_autosave_path`, `_load_autosave`, `_save_autosave`, `_get_required_factors`, `_uniquify_column_names` from `annotate_questions.py`
2. Use the same constants: `RANDOM_SEED=42`, `NUM_QUESTIONS_TO_SAMPLE=10`, same file/DB paths
3. For each question, print full context then prompt for each annotation factor
4. Autosave after each question, write final output to `claude_annotated_questions.json`
5. On restart, detect autosave and resume from first unannotated question

**Display per question:**
- Progress bar (text-based)
- Database ID, question text, evidence
- Style (value + description from STYLE_DESCRIPTIONS)
- Category (name, subname, type, definition, examples)
- Database schema with example rows (formatted as text tables)
- SQL + query results (or feedback/hidden knowledge for unanswerable)
- Disambiguation info for ambiguous questions

**Annotation prompts (numbered choices):**
- question_type: [1] Answerable [2] Ambiguous [3] Unanswerable
- question_realistic: [1] Realistic [2] Not Realistic
- category: [1] Correct [2] Incorrect
- style: [1] Correct [2] Incorrect
- sql_correct (or feedback): [1] Correct [2] Incorrect
- disambiguation (only for ambiguous+SQL): [1] Correct [2] Incorrect

**Step 2: Test manually**

Run: `python annotate_questions_cli.py`
Expected: Shows first question with full context, prompts for annotations.

**Step 3: Commit**

```bash
git add annotate_questions_cli.py
git commit -m "feat: add CLI annotation tool for question quality assessment"
```
