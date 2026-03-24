"""Checkpoint utilities for resuming the generation pipeline.

Handles saving/loading of intermediate question results and validation
tracking stages so the pipeline can resume from the last completed step.
"""

import json
import logging
import os

from dataset_dataclasses.question import Question, QuestionUnanswerable
from dataset_dataclasses.council_tracking import (
    ValidationStageResult,
    ModelVote,
    QuestionVotes,
)

logger = logging.getLogger(__name__)


def load_questions(path: str) -> list[Question]:
    """Load questions from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions: list[Question] = []
    for d in data:
        if "hidden_knowledge" in d:
            questions.append(QuestionUnanswerable.from_dict(d))
        else:
            questions.append(Question.from_dict(d))
    return questions


def save_questions(questions: list[Question], folder: str, label: str) -> None:
    """Save questions to an intermediate results JSON file."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"intermediate_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([q.to_dict() for q in questions], f, ensure_ascii=False, indent=4)


def save_tracking(stages: list[ValidationStageResult], folder: str) -> None:
    """Save validation tracking stages to disk."""
    os.makedirs(folder, exist_ok=True)
    from dataclasses import asdict
    path = os.path.join(folder, "tracking_stages.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in stages], f, indent=2, ensure_ascii=False)


def load_tracking(folder: str | None) -> list[ValidationStageResult]:
    """Load previously saved tracking stages from disk."""
    if folder is None:
        return []
    path = os.path.join(folder, "tracking_stages.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stages = []
        for s in data:
            qvotes = []
            for qv in s.get("question_votes", []):
                votes = [ModelVote(**v) for v in qv["votes"]]
                qvotes.append(QuestionVotes(
                    question_index=qv["question_index"],
                    question_text=qv["question_text"],
                    votes=votes,
                    aggregate_result=qv["aggregate_result"],
                    removed=qv["removed"],
                ))
            stages.append(ValidationStageResult(
                stage_name=s["stage_name"],
                validities=s["validities"],
                question_votes=qvotes,
            ))
        return stages
    except Exception as e:
        logger.warning("Failed to load tracking stages: %s", e)
        return []


def load_checkpoint(folder: str | None, label: str) -> list[Question] | None:
    """Try to load questions from a checkpoint file. Returns None if not found."""
    if folder is None:
        return None
    path = os.path.join(folder, f"intermediate_{label}.json")
    if not os.path.isfile(path):
        return None
    try:
        questions = load_questions(path)
        logger.info("Resumed from checkpoint '%s' (%d questions)", label, len(questions))
        return questions
    except Exception as e:
        logger.warning("Failed to load checkpoint '%s': %s", label, e)
        return None
