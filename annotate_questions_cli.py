#!/usr/bin/env python3
"""CLI tool for annotating question quality, mirroring the Streamlit annotation app."""

import json
import sys
import os

from annotate_questions import (
    load_questions,
    sample_questions,
    execute_sql_query,
    get_database_schema_with_examples,
    _get_autosave_path,
    _load_autosave,
    _save_autosave,
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

OUTPUT_FILE_PATH = "claude_annotated_questions.json"
AUTOSAVE_PREFIX = "cli"


def print_separator(char="=", width=80):
    print(char * width)


def print_header(text, char="=", width=80):
    print()
    print_separator(char, width)
    print(f"  {text}")
    print_separator(char, width)


def print_subheader(text):
    print(f"\n--- {text} ---")


def print_progress(current, total, completed):
    bar_width = 40
    filled = int(bar_width * completed / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\n[{bar}] {completed}/{total} annotated  |  Question {current + 1}/{total}")


def format_table(columns, rows, max_col_width=30):
    """Format rows as a simple text table."""
    if not columns:
        return "(no columns)"
    if not rows:
        return "(no rows)"

    str_rows = []
    for row in rows:
        str_rows.append([str(v)[:max_col_width] for v in row])

    col_widths = []
    for i, col in enumerate(columns):
        max_w = len(str(col))
        for row in str_rows:
            if i < len(row):
                max_w = max(max_w, len(row[i]))
        col_widths.append(min(max_w, max_col_width))

    header = " | ".join(str(c).ljust(w) for c, w in zip(columns, col_widths))
    divider = "-+-".join("-" * w for w in col_widths)
    lines = [header, divider]
    for row in str_rows:
        line = " | ".join(
            (row[i] if i < len(row) else "").ljust(col_widths[i])
            for i in range(len(columns))
        )
        lines.append(line)
    return "\n".join(lines)


def display_schema(db_dataset, db_id):
    """Display database schema with example rows."""
    schema_info = get_database_schema_with_examples(db_dataset, db_id)
    if not schema_info:
        print("  (Could not load database schema)")
        return

    for table_name, table_data in schema_info.items():
        print(f"\n  Table: {table_name}")
        cols_text = ", ".join(f"{c[0]} ({c[1]})" for c in table_data["columns"])
        print(f"  Columns: {cols_text}")

        if table_data.get("foreign_keys"):
            for fk in table_data["foreign_keys"]:
                print(f"    FK: {fk['from_column']} -> {fk['to_table']}.{fk['to_column']}")

        if table_data["example"]:
            col_names = [c[0] for c in table_data["columns"]]
            uniq = _uniquify_column_names(col_names)
            table_str = format_table(uniq, table_data["example"])
            for line in table_str.split("\n"):
                print(f"    {line}")
        else:
            print("    (no data)")


def display_question(question: Question, idx: int, total: int, completed: int, db_dataset=None):
    """Display all context for a question."""
    print_progress(idx, total, completed)
    print_header(f"Question {idx + 1} of {total}")

    # Basic info
    print(f"\n  Database ID: {question.db_id}")
    print(f"\n  Question:")
    print(f"    {question.question}")

    if question.evidence:
        print(f"\n  Evidence:")
        print(f"    {question.evidence}")

    # Style
    print_subheader("Style")
    print(f"  Assigned: {question.question_style.value}")
    style_desc = STYLE_DESCRIPTIONS[question.question_style]
    for line in style_desc.strip().split("\n"):
        print(f"    {line}")

    # Category
    print_subheader("Category Information")
    print(f"  Name: {question.category.get_name()}")
    if question.category.get_subname():
        print(f"  Subname: {question.category.get_subname()}")

    if question.category.is_answerable():
        cat_type = "answerable"
    elif question.category.is_solvable():
        cat_type = "ambiguous"
    else:
        cat_type = "unanswerable"
    print(f"  Type: {cat_type}")

    print(f"\n  Definition:")
    # Wrap definition text
    definition = question.category.get_definition()
    for line in _wrap_text(definition, 76):
        print(f"    {line}")

    examples = question.category.get_examples()
    if examples:
        print(f"\n  Examples:")
        for ex in examples:
            print(f"    - {ex}")

    # Database schema
    if db_dataset:
        print_subheader("Database Schema (up to 3 example rows per table)")
        display_schema(db_dataset, question.db_id)

    # Disambiguation info for ambiguous questions
    is_ambiguous = not question.category.is_answerable() and question.category.is_solvable()

    if is_ambiguous and isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        print_subheader("Disambiguation / Hidden Knowledge")
        print(f"    {question.hidden_knowledge}")

    # SQL and results, or feedback
    if question.sql:
        print_subheader("SQL Query")
        print(f"    {question.sql}")

        if db_dataset:
            columns, result = execute_sql_query(db_dataset, question.db_id, question.sql, limit=5)
            if columns is not None and isinstance(result, list):
                print(f"\n  Query Results (up to 5 rows):")
                if result:
                    uniq_cols = _uniquify_column_names(columns)
                    table_str = format_table(uniq_cols, result)
                    for line in table_str.split("\n"):
                        print(f"    {line}")
                else:
                    print("    (query returned no results)")
            else:
                print(f"    Query error: {result}")
    else:
        if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
            print_subheader("Feedback / Hidden Knowledge")
            print(f"    {question.hidden_knowledge}")


def _wrap_text(text, width=76):
    """Simple word-wrap."""
    words = text.split()
    lines = []
    current = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return lines


def prompt_choice(label, options):
    """Prompt user for a numbered choice. Returns the selected value."""
    print(f"\n  {label}")
    for i, (display, value) in enumerate(options, 1):
        print(f"    [{i}] {display}")

    while True:
        try:
            raw = input("  > ").strip()
            if not raw:
                continue
            choice = int(raw)
            if 1 <= choice <= len(options):
                return options[choice - 1][1]
            print(f"    Please enter a number between 1 and {len(options)}")
        except ValueError:
            print(f"    Please enter a number between 1 and {len(options)}")
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted. Progress has been autosaved.")
            sys.exit(0)


def annotate_question(question: Question):
    """Collect all annotations for a question. Returns a dict of factor -> value."""
    annotations = {}

    print_subheader("Annotations")

    # Question type
    annotations["question_type"] = prompt_choice(
        "What type of question is this?",
        [("Answerable", "Answerable"), ("Ambiguous", "Ambiguous"), ("Unanswerable", "Unanswerable")],
    )

    # Question realistic
    annotations["question_realistic"] = prompt_choice(
        "Is the question realistic?",
        [("Realistic", True), ("Not Realistic", False)],
    )

    # Category correct
    annotations["category"] = prompt_choice(
        "Is the assigned category correct?",
        [("Correct", True), ("Incorrect", False)],
    )

    # Style correct
    annotations["style"] = prompt_choice(
        "Does the style match?",
        [("Correct", True), ("Incorrect", False)],
    )

    # Disambiguation (only for ambiguous questions with SQL)
    is_ambiguous = not question.category.is_answerable() and question.category.is_solvable()
    if is_ambiguous and question.sql:
        annotations["disambiguation"] = prompt_choice(
            "Is the disambiguation information correct?",
            [("Correct", True), ("Incorrect", False)],
        )

    # SQL / Feedback correct
    if question.sql:
        annotations["sql_correct"] = prompt_choice(
            "Is the SQL semantically correct?",
            [("Correct", True), ("Incorrect", False)],
        )
    else:
        annotations["sql_correct"] = prompt_choice(
            "Is the feedback correct and actionable?",
            [("Correct", True), ("Incorrect", False)],
        )

    return annotations


def get_completion_count(all_annotations):
    """Count how many questions are fully annotated."""
    return sum(
        1 for ann in all_annotations.values()
        if all(v is not None for v in ann.values())
    )


def main():
    print_header("Question Quality Annotation Tool (CLI)")
    print(f"  Questions file: {QUESTIONS_FILE_PATH}")
    print(f"  Output file:    {OUTPUT_FILE_PATH}")
    print(f"  DB root:        {DB_ROOT_PATH}")
    print(f"  Seed: {RANDOM_SEED}  |  Samples per category: {NUM_QUESTIONS_TO_SAMPLE}")

    # Load and sample questions
    print("\nLoading questions...")
    questions = load_questions(QUESTIONS_FILE_PATH)
    sampled = sample_questions(questions, RANDOM_SEED, NUM_QUESTIONS_TO_SAMPLE)
    print(f"Sampled {len(sampled)} questions across categories.")

    # Initialize DB
    db_dataset = None
    if DB_ROOT_PATH and DB_NAME:
        try:
            db_dataset = DBDataset(DB_ROOT_PATH, DB_NAME)
            print(f"Database loaded: {DB_NAME}")
        except Exception as e:
            print(f"Warning: Could not load database: {e}")

    # Autosave setup
    autosave_path = _get_autosave_path(QUESTIONS_FILE_PATH, RANDOM_SEED, NUM_QUESTIONS_TO_SAMPLE)
    # Use a different autosave file for CLI to avoid conflicts with Streamlit
    autosave_path = autosave_path.replace("autosave_", f"{AUTOSAVE_PREFIX}_autosave_")

    # Build required factors per question
    required_per_question = {
        i: set(_get_required_factors(q)) for i, q in enumerate(sampled)
    }

    # Try to restore from autosave
    saved = _load_autosave(autosave_path)
    if saved is not None and len(saved) == len(sampled):
        all_annotations = saved
        # Ensure all required factors exist
        for i in range(len(sampled)):
            required = required_per_question[i]
            entry = all_annotations.get(i, {})
            for factor in required:
                if factor not in entry:
                    entry[factor] = None
            for key in list(entry.keys()):
                if key not in required:
                    del entry[key]
            all_annotations[i] = entry
        completed = get_completion_count(all_annotations)
        print(f"Restored autosave: {completed}/{len(sampled)} already annotated.")
    else:
        all_annotations = {
            i: {factor: None for factor in required_per_question[i]}
            for i in range(len(sampled))
        }

    # Find first unannotated question
    start_idx = 0
    for i in range(len(sampled)):
        if all(v is not None for v in all_annotations[i].values()):
            start_idx = i + 1
        else:
            start_idx = i
            break

    # Main annotation loop
    for idx in range(start_idx, len(sampled)):
        question = sampled[idx]
        completed = get_completion_count(all_annotations)

        # Skip already completed
        if all(v is not None for v in all_annotations[idx].values()):
            continue

        display_question(question, idx, len(sampled), completed, db_dataset)
        annotations = annotate_question(question)
        all_annotations[idx] = annotations

        # Autosave
        _save_autosave(autosave_path, all_annotations)

        completed = get_completion_count(all_annotations)
        print(f"\n  Saved. Progress: {completed}/{len(sampled)}")

    # Final save
    print_header("Annotation Complete!")
    completed = get_completion_count(all_annotations)
    print(f"  {completed}/{len(sampled)} questions annotated.")

    results = []
    for i, question in enumerate(sampled):
        results.append({
            "question": question.to_dict(),
            "quality_annotations": all_annotations[i],
        })

    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Results saved to {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()
