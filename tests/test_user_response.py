"""Tests for two-stage UserResponse orchestrator."""
import pytest
from unittest.mock import MagicMock, patch
from dataset_dataclasses.benchmark import Conversation, Interaction, SystemResponse, RelevancyLabel, CategoryUse
from dataset_dataclasses.question import Question, QuestionUnanswerable, QuestionStyle, QuestionDifficulty
from users.user_response import UserResponse


def _make_category(is_answerable=False, is_solvable=True):
    cat = MagicMock()
    cat.is_answerable.return_value = is_answerable
    cat.is_solvable.return_value = is_solvable
    return cat


def _make_unsolvable_conv():
    cat = _make_category(is_answerable=False, is_solvable=False)
    q = QuestionUnanswerable(
        db_id="test_db", question="Q", evidence=None, sql=None,
        category=cat, question_style=QuestionStyle.FORMAL,
        question_difficulty=QuestionDifficulty.SIMPLE,
        hidden_knowledge="Missing data", is_solvable=False,
    )
    return Conversation(
        question=q,
        interactions=[Interaction(system_response=SystemResponse(system_question="What?"))],
        category_use=CategoryUse.GROUND_TRUTH,
    )


class TestShortCircuitUnsolvable:
    def test_unsolvable_gets_irrelevant_and_canned_refusal(self):
        db = MagicMock()
        model = MagicMock()
        model.model_name = "test-model"
        ur = UserResponse(db, [model])

        conv = _make_unsolvable_conv()
        ur.get_response([conv])

        assert conv.interactions[-1].relevance == RelevancyLabel.IRRELEVANT
        assert conv.interactions[-1].user_response == "That's not relevant to my question."
        model.init.assert_not_called()


def _make_solvable_conv(system_question="What do you mean by 'recent'?"):
    cat = _make_category(is_answerable=False, is_solvable=True)
    q = QuestionUnanswerable(
        db_id="test_db", question="Show me recent students", evidence=None,
        sql="SELECT name FROM students ORDER BY enrollment_date DESC LIMIT 10",
        category=cat, question_style=QuestionStyle.COLLOQUIAL,
        question_difficulty=QuestionDifficulty.SIMPLE,
        hidden_knowledge="Recent means enrolled in the last 6 months",
        is_solvable=True,
    )
    return Conversation(
        question=q,
        interactions=[Interaction(system_response=SystemResponse(system_question=system_question))],
        category_use=CategoryUse.GROUND_TRUTH,
    )


def _make_answerable_conv(system_question="How many results do you want?"):
    cat = _make_category(is_answerable=True, is_solvable=True)
    q = Question(
        db_id="test_db", question="List all students", evidence=None,
        sql="SELECT name FROM students ORDER BY name ASC LIMIT 20",
        category=cat, question_style=QuestionStyle.FORMAL,
        question_difficulty=QuestionDifficulty.SIMPLE,
    )
    return Conversation(
        question=q,
        interactions=[Interaction(system_response=SystemResponse(system_question=system_question))],
        category_use=CategoryUse.GROUND_TRUTH,
    )


def _mock_model(name, classify_response, answer_response):
    """Create a mock model that returns classify_response for Stage 1 and answer_response for Stage 2."""
    model = MagicMock()
    model.model_name = name
    call_count = [0]

    def mock_batch(prompts, model_classes):
        call_count[0] += 1
        if call_count[0] == 1:
            return [classify_response] * len(prompts)
        else:
            return [answer_response] * len(prompts)

    model.generate_batch_with_constraints = MagicMock(side_effect=mock_batch)
    return model


class TestTwoStageFlowSolvable:
    def test_solvable_relevant_majority(self):
        """Three models vote: 2 RELEVANT + 1 TECHNICAL -> RELEVANT wins."""
        from users.prompts.user_classify_prompt import UserClassifySolvable
        from users.prompts.user_answer_prompt import UserAnswerModel

        db = MagicMock()
        relevant_classify = UserClassifySolvable(relevancy="Relevant", node_ids=[])
        technical_classify = UserClassifySolvable(relevancy="Technical", node_ids=[1])
        answer = UserAnswerModel(answer="I mean the last 6 months")

        m1 = _mock_model("m1", relevant_classify, answer)
        m2 = _mock_model("m2", relevant_classify, answer)
        m3 = _mock_model("m3", technical_classify, answer)

        ur = UserResponse(db, [m1, m2, m3])
        conv = _make_solvable_conv()
        ur.get_response([conv])

        assert conv.interactions[-1].relevance == RelevancyLabel.RELEVANT
        assert conv.interactions[-1].user_response == "I mean the last 6 months"

    def test_solvable_technical_unions_node_ids(self):
        """Two models vote TECHNICAL with different nodes -> union taken."""
        from users.prompts.user_classify_prompt import UserClassifySolvable
        from users.prompts.user_answer_prompt import UserAnswerModel

        db = MagicMock()
        tech1 = UserClassifySolvable(relevancy="Technical", node_ids=[1, 2])
        tech2 = UserClassifySolvable(relevancy="Technical", node_ids=[2, 3])
        answer = UserAnswerModel(answer="Sort by newest first, top 10")

        m1 = _mock_model("m1", tech1, answer)
        m2 = _mock_model("m2", tech2, answer)

        ur = UserResponse(db, [m1, m2])
        conv = _make_solvable_conv()
        ur.get_response([conv])

        assert conv.interactions[-1].relevance == RelevancyLabel.TECHNICAL
        assert conv.interactions[-1].user_response == "Sort by newest first, top 10"

    def test_tie_resolves_to_irrelevant(self):
        """1 RELEVANT + 1 TECHNICAL + 1 IRRELEVANT -> tie -> IRRELEVANT."""
        from users.prompts.user_classify_prompt import UserClassifySolvable
        from users.prompts.user_answer_prompt import UserAnswerModel

        db = MagicMock()
        r = UserClassifySolvable(relevancy="Relevant", node_ids=[])
        t = UserClassifySolvable(relevancy="Technical", node_ids=[1])
        ir = UserClassifySolvable(relevancy="Irrelevant", node_ids=[])
        answer = UserAnswerModel(answer="Not relevant")

        m1 = _mock_model("m1", r, answer)
        m2 = _mock_model("m2", t, answer)
        m3 = _mock_model("m3", ir, answer)

        ur = UserResponse(db, [m1, m2, m3])
        conv = _make_solvable_conv()
        ur.get_response([conv])

        assert conv.interactions[-1].relevance == RelevancyLabel.IRRELEVANT


class TestTwoStageFlowAnswerable:
    def test_answerable_technical(self):
        """Answerable question classified as TECHNICAL."""
        from users.prompts.user_classify_prompt import UserClassifyAnswerable
        from users.prompts.user_answer_prompt import UserAnswerModel

        db = MagicMock()
        classify = UserClassifyAnswerable(relevancy="Technical", node_ids=[1])
        answer = UserAnswerModel(answer="I'd like 20 results please")

        m1 = _mock_model("m1", classify, answer)
        ur = UserResponse(db, [m1])
        conv = _make_answerable_conv()
        ur.get_response([conv])

        assert conv.interactions[-1].relevance == RelevancyLabel.TECHNICAL
        assert conv.interactions[-1].user_response == "I'd like 20 results please"


class TestSingleModelSkipsTournament:
    def test_single_model_no_tournament(self):
        """With 1 model, tournament is skipped and answer is used directly."""
        from users.prompts.user_classify_prompt import UserClassifySolvable
        from users.prompts.user_answer_prompt import UserAnswerModel

        db = MagicMock()
        classify = UserClassifySolvable(relevancy="Relevant", node_ids=[])
        answer = UserAnswerModel(answer="I mean recent as in last 6 months")

        m1 = _mock_model("m1", classify, answer)
        ur = UserResponse(db, [m1])
        conv = _make_solvable_conv()
        relevancy_tracking, tournament_tracking = ur.get_response([conv])

        assert conv.interactions[-1].user_response == "I mean recent as in last 6 months"
        assert tournament_tracking == []
