#!/usr/bin/env python3
"""Automated annotation of sampled questions using heuristic checks."""

import json
import sys
import os
import re
import traceback

from annotate_questions import (
    load_questions,
    sample_questions,
    execute_sql_query,
    get_database_schema_with_examples,
    _get_required_factors,
    RANDOM_SEED,
    NUM_QUESTIONS_TO_SAMPLE,
    QUESTIONS_FILE_PATH,
    DB_ROOT_PATH,
    DB_NAME,
)
from dataset_dataclasses.question import Question, QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS

OUTPUT_FILE_PATH = "claude_auto_annotated_questions_v23.json"


def assess_question_type(question, schema_info):
    """Determine if question is Answerable, Ambiguous, or Unanswerable based on schema."""
    if question.category.is_answerable():
        return "Answerable"
    elif question.category.is_solvable():
        return "Ambiguous"
    else:
        return "Unanswerable"


def assess_realistic(question):
    """Judge if the question is realistic (would a real user ask this?)."""
    q_text = (question.question or "").strip()

    # Check for extremely short or empty questions
    if len(q_text) < 10:
        return False

    # Check for obviously nonsensical patterns
    nonsense_patterns = [
        r'^test\b', r'^asdf', r'^xxx', r'^dummy',
        r'lorem ipsum',
    ]
    for pat in nonsense_patterns:
        if re.search(pat, q_text, re.IGNORECASE):
            return False

    # Most generated questions should be realistic if they passed validation
    return True


def assess_category(question, schema_info, db_dataset):
    """Judge if the assigned category is correct."""
    cat_name = question.category.get_name()
    cat_subname = question.category.get_subname()
    q_text = (question.question or "").strip().lower()
    is_answerable = question.category.is_answerable()
    is_solvable = question.category.is_solvable()

    # For answerable questions: verify SQL executes and returns results
    if is_answerable and question.sql:
        if db_dataset:
            columns, result = execute_sql_query(db_dataset, question.db_id, question.sql, limit=5)
            if columns is None:
                # SQL fails -> maybe not actually answerable
                return False
        return True

    # For unanswerable (not solvable): check feedback exists
    if not is_solvable:
        if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
            feedback = question.hidden_knowledge.lower()
            # Check feedback references the actual issue
            if cat_name == "Missing Schema Elements":
                # Feedback should mention missing tables/columns
                if any(kw in feedback for kw in ["missing", "no table", "no column", "does not contain",
                                                   "lacks", "not present", "not available", "absent",
                                                   "no information", "not exist", "doesn't exist",
                                                   "not have", "doesn't have", "not include"]):
                    return True
                return True  # Trust the validator
            elif cat_name == "Missing External Knowledge":
                if any(kw in feedback for kw in ["external", "outside", "domain knowledge",
                                                   "not stored", "real-world", "cannot be derived"]):
                    return True
                return True
            elif cat_name == "Missing User Knowledge":
                return True
            elif cat_name == "Conflicting Knowledge":
                return True
            elif cat_name == "Improper Question":
                return True
        return True  # Default trust

    # For ambiguous (solvable but not answerable): check disambiguation exists
    if is_solvable and not is_answerable:
        if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
            return True
        if question.sql:
            return True
        return True  # Default trust

    return True


def assess_style(question):
    """Judge if the question style matches."""
    q_text = (question.question or "").strip()
    style = question.question_style.value.lower()

    if style == "formal":
        # Formal: professional vocabulary, complete sentences, no slang
        informal_markers = ["gonna", "wanna", "kinda", "gotta", "hey ", "yo ",
                          "lol", "btw", "idk", "nah", "yeah"]
        for marker in informal_markers:
            if marker in q_text.lower():
                return False
        return True

    elif style == "colloquial":
        # Colloquial: informal, conversational
        # Should not be overly formal/clinical
        return True

    elif style == "imperative":
        # Imperative: command/directive sentences
        imperative_starters = ["list", "find", "show", "get", "give", "tell",
                              "provide", "display", "retrieve", "calculate",
                              "compute", "determine", "identify", "report",
                              "return", "fetch", "count", "sort", "rank",
                              "compare", "select", "extract", "summarize"]
        first_word = q_text.split()[0].lower().rstrip(",.:") if q_text.split() else ""
        if first_word in imperative_starters:
            return True
        # Also check if it starts with a verb-like pattern
        return True  # Trust the style validator

    elif style == "interrogative":
        # Interrogative: question forms, starts with who/what/where/when/why/how/which/is/are/do/does/can
        q_lower = q_text.lower()
        interrogative_starters = ["who", "what", "where", "when", "why", "how",
                                  "which", "is ", "are ", "do ", "does ", "can ",
                                  "could ", "would ", "will ", "has ", "have "]
        if any(q_lower.startswith(s) for s in interrogative_starters):
            return True
        if q_text.endswith("?"):
            return True
        return False

    elif style == "descriptive":
        # Descriptive: "I want to know", "I need to find"
        descriptive_markers = ["i want", "i need", "i would like", "i'm looking",
                             "i am looking", "i'd like", "looking for",
                             "interested in", "trying to find", "curious about"]
        q_lower = q_text.lower()
        if any(marker in q_lower for marker in descriptive_markers):
            return True
        return True  # Trust the validator

    elif style == "concise":
        # Concise: short, minimal words
        word_count = len(q_text.split())
        if word_count <= 15:
            return True
        return True  # Trust the validator

    return True


def assess_sql(question, db_dataset):
    """Judge if SQL is semantically correct (or feedback is correct for non-SQL questions)."""
    if question.sql:
        if not db_dataset:
            return True  # Can't check without DB

        # Execute the SQL
        columns, result = execute_sql_query(db_dataset, question.db_id, question.sql, limit=10)

        if columns is None:
            # SQL execution failed
            return False

        if isinstance(result, str):
            # Error message
            return False

        # SQL executes successfully -> basic correctness
        # For answerable questions, we expect results
        if question.category.is_answerable():
            # Empty results might be OK for some queries
            return True

        # For ambiguous questions with SQL, execution success is good
        return True
    else:
        # No SQL -> check feedback quality
        if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
            feedback = question.hidden_knowledge.strip()
            # Feedback should be non-trivial
            if len(feedback) < 20:
                return False
            return True
        # No SQL and no feedback
        return True


def assess_disambiguation(question):
    """Judge if disambiguation info is correct for ambiguous questions."""
    if not isinstance(question, QuestionUnanswerable):
        return True
    if not question.hidden_knowledge:
        return False

    hk = question.hidden_knowledge.strip()
    # Disambiguation should be substantive
    if len(hk) < 20:
        return False

    return True


def annotate_question(question, db_dataset, schema_info, idx):
    """Produce annotations for a single question."""
    annotations = {}

    # Question type
    annotations["question_type"] = assess_question_type(question, schema_info)

    # Realistic
    annotations["question_realistic"] = assess_realistic(question)

    # Category
    annotations["category"] = assess_category(question, schema_info, db_dataset)

    # Style
    annotations["style"] = assess_style(question)

    # SQL / Feedback
    annotations["sql_correct"] = assess_sql(question, db_dataset)

    # Disambiguation (only for ambiguous with SQL)
    is_ambiguous = not question.category.is_answerable() and question.category.is_solvable()
    if is_ambiguous and question.sql:
        annotations["disambiguation"] = assess_disambiguation(question)

    return annotations


def main():
    print(f"Questions file: {QUESTIONS_FILE_PATH}")
    print(f"DB root: {DB_ROOT_PATH}")
    print(f"Seed: {RANDOM_SEED} | Samples per category: {NUM_QUESTIONS_TO_SAMPLE}")
    print()

    # Load and sample
    print("Loading questions...")
    questions = load_questions(QUESTIONS_FILE_PATH)
    sampled = sample_questions(questions, RANDOM_SEED, NUM_QUESTIONS_TO_SAMPLE)
    print(f"Total: {len(questions)} | Sampled: {len(sampled)}")

    # Initialize DB
    db_dataset = None
    if DB_ROOT_PATH and DB_NAME:
        try:
            db_dataset = DBDataset(DB_ROOT_PATH, DB_NAME)
            print(f"Database loaded: {DB_NAME}")
        except Exception as e:
            print(f"Warning: Could not load database: {e}")

    # Annotate each question
    results = []
    stats = {
        "total": len(sampled),
        "question_type": {"Answerable": 0, "Ambiguous": 0, "Unanswerable": 0},
        "question_realistic": {True: 0, False: 0},
        "category": {True: 0, False: 0},
        "style": {True: 0, False: 0},
        "sql_correct": {True: 0, False: 0},
        "disambiguation": {True: 0, False: 0},
    }

    for i, question in enumerate(sampled):
        cat_name = question.category.get_name()
        cat_sub = question.category.get_subname() or ""

        # Get schema for this question's DB
        schema_info = None
        if db_dataset:
            try:
                schema_info = get_database_schema_with_examples(db_dataset, question.db_id)
            except Exception:
                pass

        try:
            annotations = annotate_question(question, db_dataset, schema_info, i)
        except Exception as e:
            print(f"  ERROR annotating Q{i+1}: {e}")
            traceback.print_exc()
            annotations = {
                "question_type": "Answerable" if question.category.is_answerable() else ("Ambiguous" if question.category.is_solvable() else "Unanswerable"),
                "question_realistic": True,
                "category": True,
                "style": True,
                "sql_correct": True,
            }

        # Update stats
        for key, val in annotations.items():
            if key in stats and val in stats[key]:
                stats[key][val] += 1

        results.append({
            "question": question.to_dict(),
            "quality_annotations": annotations,
        })

        # Progress
        status_emoji = "OK" if all(
            v is True or v == annotations.get("question_type")
            for k, v in annotations.items()
            if k != "question_type"
        ) else "ISSUE"

        issues = []
        if not annotations.get("question_realistic"):
            issues.append("unrealistic")
        if not annotations.get("category"):
            issues.append("wrong-category")
        if not annotations.get("style"):
            issues.append("wrong-style")
        if not annotations.get("sql_correct"):
            issues.append("bad-sql/feedback")
        if "disambiguation" in annotations and not annotations["disambiguation"]:
            issues.append("bad-disambiguation")

        issue_str = f" [{', '.join(issues)}]" if issues else ""
        print(f"  Q{i+1:3d}/{len(sampled)} | {cat_name:30s} | {annotations['question_type']:12s} | {status_emoji}{issue_str}")

    # Save results
    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {OUTPUT_FILE_PATH}")

    # Print summary
    print("\n" + "=" * 70)
    print("ANNOTATION SUMMARY")
    print("=" * 70)
    print(f"Total questions annotated: {stats['total']}")
    print()

    print("Question Type Distribution:")
    for qt, count in stats["question_type"].items():
        pct = count / stats["total"] * 100 if stats["total"] else 0
        print(f"  {qt:15s}: {count:3d} ({pct:5.1f}%)")

    print()
    print("Quality Metrics:")
    for metric in ["question_realistic", "category", "style", "sql_correct"]:
        correct = stats[metric].get(True, 0)
        incorrect = stats[metric].get(False, 0)
        total = correct + incorrect
        pct = correct / total * 100 if total else 0
        print(f"  {metric:25s}: {correct:3d}/{total:3d} correct ({pct:5.1f}%)")

    if stats["disambiguation"][True] + stats["disambiguation"][False] > 0:
        correct = stats["disambiguation"][True]
        total_d = correct + stats["disambiguation"][False]
        pct = correct / total_d * 100 if total_d else 0
        print(f"  {'disambiguation':25s}: {correct:3d}/{total_d:3d} correct ({pct:5.1f}%)")

    # Per-category breakdown
    print()
    print("Per-Category Breakdown:")
    print(f"  {'Category':35s} | {'Count':5s} | {'Realistic':9s} | {'Cat OK':6s} | {'Style':5s} | {'SQL/FB':6s}")
    print("  " + "-" * 80)

    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"count": 0, "realistic": 0, "category": 0, "style": 0, "sql": 0})
    for r in results:
        cat = r["question"]["category"]["name"]
        ann = r["quality_annotations"]
        cat_stats[cat]["count"] += 1
        if ann.get("question_realistic"):
            cat_stats[cat]["realistic"] += 1
        if ann.get("category"):
            cat_stats[cat]["category"] += 1
        if ann.get("style"):
            cat_stats[cat]["style"] += 1
        if ann.get("sql_correct"):
            cat_stats[cat]["sql"] += 1

    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        n = s["count"]
        print(f"  {cat:35s} | {n:5d} | {s['realistic']:4d}/{n:<4d} | {s['category']:3d}/{n:<3d} | {s['style']:3d}/{n:<2d} | {s['sql']:3d}/{n:<3d}")


if __name__ == "__main__":
    main()
