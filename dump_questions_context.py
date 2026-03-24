#!/usr/bin/env python3
"""Dump full context for all sampled questions for manual review."""

import json
import sys
import os

from annotate_questions import (
    load_questions,
    sample_questions,
    execute_sql_query,
    get_database_schema_with_examples,
    _get_required_factors,
    _uniquify_column_names,
    RANDOM_SEED,
    NUM_QUESTIONS_TO_SAMPLE,
    QUESTIONS_FILE_PATH,
    DB_ROOT_PATH,
    DB_NAME,
)
from dataset_dataclasses.question import Question, QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS


def dump_question(question, idx, total, db_dataset):
    """Dump full context for a question."""
    out = {}
    out["index"] = idx
    out["db_id"] = question.db_id
    out["question_text"] = question.question
    out["evidence"] = question.evidence

    # Category
    out["category_name"] = question.category.get_name()
    out["category_subname"] = question.category.get_subname()
    out["category_definition"] = question.category.get_definition()
    out["category_examples"] = question.category.get_examples()
    out["is_answerable"] = question.category.is_answerable()
    out["is_solvable"] = question.category.is_solvable()

    if out["is_answerable"]:
        out["category_type"] = "answerable"
    elif out["is_solvable"]:
        out["category_type"] = "ambiguous"
    else:
        out["category_type"] = "unanswerable"

    # Style
    out["style"] = question.question_style.value
    out["style_definition"] = STYLE_DESCRIPTIONS[question.question_style]

    # Hidden knowledge / feedback
    if isinstance(question, QuestionUnanswerable):
        out["hidden_knowledge"] = question.hidden_knowledge
    else:
        out["hidden_knowledge"] = None

    # SQL
    out["sql"] = question.sql

    # SQL execution results
    if question.sql and db_dataset:
        columns, result = execute_sql_query(db_dataset, question.db_id, question.sql, limit=5)
        if columns is not None and isinstance(result, list):
            out["sql_columns"] = columns
            out["sql_results"] = [list(row) for row in result]
            out["sql_error"] = None
        else:
            out["sql_columns"] = None
            out["sql_results"] = None
            out["sql_error"] = str(result)
    else:
        out["sql_columns"] = None
        out["sql_results"] = None
        out["sql_error"] = None

    # Schema (compact: just table names, columns, and FKs)
    if db_dataset:
        schema_info = get_database_schema_with_examples(db_dataset, question.db_id)
        if schema_info:
            compact_schema = {}
            for table_name, table_data in schema_info.items():
                compact_schema[table_name] = {
                    "columns": [f"{c[0]} ({c[1]})" for c in table_data["columns"]],
                    "foreign_keys": table_data.get("foreign_keys", []),
                    "example_rows": [list(row) for row in table_data["example"][:2]] if table_data["example"] else []
                }
            out["schema"] = compact_schema
        else:
            out["schema"] = None
    else:
        out["schema"] = None

    # Required factors
    out["required_factors"] = _get_required_factors(question)

    return out


def main():
    print("Loading questions...", file=sys.stderr)
    questions = load_questions(QUESTIONS_FILE_PATH)
    sampled = sample_questions(questions, RANDOM_SEED, NUM_QUESTIONS_TO_SAMPLE)
    print(f"Sampled {len(sampled)} questions", file=sys.stderr)

    db_dataset = None
    if DB_ROOT_PATH and DB_NAME:
        db_dataset = DBDataset(DB_ROOT_PATH, DB_NAME)
        print(f"Database loaded: {DB_NAME}", file=sys.stderr)

    all_contexts = []
    for i, q in enumerate(sampled):
        ctx = dump_question(q, i, len(sampled), db_dataset)
        all_contexts.append(ctx)
        print(f"  Dumped Q{i+1}/{len(sampled)}", file=sys.stderr)

    with open("question_contexts_v23.json", "w") as f:
        json.dump(all_contexts, f, indent=2)
    print(f"\nDumped {len(all_contexts)} questions to question_contexts_v23.json", file=sys.stderr)


if __name__ == "__main__":
    main()
