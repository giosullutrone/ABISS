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
