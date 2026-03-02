from validators.validator import Validator
from validators.sql_executability import SQLExecutability
from validators.gt_satisfaction import GTSatisfaction
from dataset_dataclasses.question import Question
from dataset_dataclasses.council_tracking import ValidationStageResult, QuestionVotes, ModelVote
from models.model import Model
from db_datasets.db_dataset import DBDataset
from copy import deepcopy


class UnsolvabilityVerification(Validator):
    """
    Validator for unsolvable questions (questions that shouldn't have an SQL answer).

    The validation process:
    1. Generate SQL attempts for unsolvable questions
    2. Apply generated SQLs to copies of the questions
    3. Check if SQLs are executable
       - If not executable -> Valid (confirmed unsolvable)
       - If executable -> Check if SQL answers the question correctly
         - If answers correctly -> Invalid (question is actually solvable)
         - If doesn't answer correctly -> Valid (confirmed unsolvable)
    """

    def __init__(self, db: DBDataset, models: list[Model], sql_executability_validator: SQLExecutability, gt_satisfaction_validator: GTSatisfaction) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.sql_executability_validator = sql_executability_validator
        self.gt_satisfaction_validator = gt_satisfaction_validator

    def validate(self, questions: list[Question]) -> ValidationStageResult:
        valids: list[bool] = [True for _ in questions]

        # Track per-model results: model_solved[question_idx][model_idx] = bool
        model_solved: dict[int, dict[int, bool]] = {i: {} for i in range(len(questions))}

        # Generate SQLs for each question using all models
        sqls: list[list[str | None]] = [self.db.generate_sqls_unsafe(model, questions) for model in self.models]

        # Create all question copies with generated SQLs at once
        all_questions_with_generated_sql: list[Question] = []
        question_mapping: list[int] = []  # Maps each generated question back to its original index
        model_mapping: list[int] = []  # Maps each generated question back to its model index

        for i, question in enumerate(questions):
            for midx, model_sqls in enumerate(sqls):
                generated_sql = model_sqls[i]
                if generated_sql is not None:  # Only validate questions with generated SQL
                    question_copy = deepcopy(question)
                    question_copy.sql = generated_sql
                    all_questions_with_generated_sql.append(question_copy)
                    question_mapping.append(i)
                    model_mapping.append(midx)

        if not all_questions_with_generated_sql:
            # No SQL was generated for any question, all remain valid (confirmed unsolvable)
            question_votes = self._build_question_votes(questions, model_solved)
            return ValidationStageResult(
                stage_name="unsolvability_verification",
                validities=valids,
                question_votes=question_votes,
            )

        # Check which generated SQLs are executable
        exec_result = self.sql_executability_validator.validate(all_questions_with_generated_sql)
        executable_flags = exec_result.validities

        # Mark non-executable as not-solved
        for j in range(len(all_questions_with_generated_sql)):
            orig_idx = question_mapping[j]
            m_idx = model_mapping[j]
            if not executable_flags[j]:
                model_solved[orig_idx][m_idx] = False

        executable_questions = [q for j, q in enumerate(all_questions_with_generated_sql) if executable_flags[j]]
        executable_orig_mapping = [question_mapping[j] for j, flag in enumerate(executable_flags) if flag]
        executable_model_mapping = [model_mapping[j] for j, flag in enumerate(executable_flags) if flag]

        if not executable_questions:
            # None of the generated SQLs are executable, all questions remain valid (confirmed unsolvable)
            question_votes = self._build_question_votes(questions, model_solved)
            return ValidationStageResult(
                stage_name="unsolvability_verification",
                validities=valids,
                question_votes=question_votes,
            )

        # For executable SQLs, check if they correctly answer the question
        gt_result = self.gt_satisfaction_validator.validate(executable_questions)
        gt_validation_flags = gt_result.validities

        # Mark questions as invalid if any of their generated SQLs are executable AND correctly answer the question
        for j, is_valid_gt in enumerate(gt_validation_flags):
            original_idx = executable_orig_mapping[j]
            m_idx = executable_model_mapping[j]
            model_solved[original_idx][m_idx] = is_valid_gt
            if is_valid_gt:
                valids[original_idx] = False

        question_votes = self._build_question_votes(questions, model_solved)
        return ValidationStageResult(
            stage_name="unsolvability_verification",
            validities=valids,
            question_votes=question_votes,
        )

    def _build_question_votes(self, questions: list[Question], model_solved: dict[int, dict[int, bool]]) -> list[QuestionVotes]:
        """Build QuestionVotes from per-model solve tracking.

        Each model's vote is True if it generated an SQL that was both executable
        and correct (meaning the question IS solvable). The aggregate_result is True
        when the question is kept (confirmed unsolvable) and False when removed
        (actually solvable).
        """
        question_votes: list[QuestionVotes] = []
        for i in range(len(questions)):
            per_model = []
            any_solved = False
            for midx in range(len(self.models)):
                solved = model_solved[i].get(midx, False)
                per_model.append(ModelVote(model_name=self.models[midx].model_name, vote=solved))
                if solved:
                    any_solved = True
            question_votes.append(QuestionVotes(
                question_index=i,
                question_text=questions[i].question,
                votes=per_model,
                aggregate_result=not any_solved,
                removed=any_solved,
            ))
        return question_votes
