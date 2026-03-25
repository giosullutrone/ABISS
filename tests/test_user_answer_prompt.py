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

    def test_empty_fragments_shows_no_preference_instruction(self):
        conv = _make_conversation()
        prompt = get_user_answer_prompt_technical(conv, [])
        assert "don't have a specific preference" in prompt.lower() or "no specific preferences" in prompt.lower()

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
