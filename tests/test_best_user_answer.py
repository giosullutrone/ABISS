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
