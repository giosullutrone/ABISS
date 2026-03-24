from dataset_dataclasses.question import Question, QuestionUnanswerable, QuestionStyle, QuestionDifficulty
from dataset_dataclasses.council_tracking import ValidationStageResult, GenerationTrackingReport
from categories.category import Category
from db_datasets.db_dataset import DBDataset
from models.model import Model
import logging
import os
from generators.prompts.generator_prompt import get_generation_prompt
from generators.prompts.sql_generation_prompt import SQLGenerationOutput, get_sql_generation_prompt
from generators.checkpoint import save_questions, save_tracking, load_tracking, load_checkpoint
from validators.difficulty_conformance import classify_sql_difficulty
from pydantic import BaseModel
from validators.sql_executability import SQLExecutability
from validators.ambiguity_verification import AmbiguityVerification
from validators.gt_satisfaction import GTSatisfaction
from validators.duplicate_removal import DuplicateRemoval
from validators.unsolvability_verification import UnsolvabilityVerification
from validators.category_consistency import CategoryConsistency
from validators.style_conformance import StyleConformance
from validators.feedback_quality_check import FeedbackQualityCheck
from validators.evidence_necessity import EvidenceNecessity
from validators.category_conformance import CategoryConformance
from categories.answerable_with_evidence import AnswerableWithEvidenceCategory
from validators.validator import Validator

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self,
                 db: DBDataset,
                 models: list[Model],
                 models_validator: list[Model],
                 categories: list[Category],
                 n_samples: int,
                 max_tokens: int,
                 max_gen_tokens: int,
                 intermediate_results_folder: str | None) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.n_samples: int = n_samples
        self.intermediate_results_folder = intermediate_results_folder

        self.sql_executability_validator = SQLExecutability(db)
        self.ambiguity_verification_validator = AmbiguityVerification(db, models_validator)
        self.gt_satisfaction_validator = GTSatisfaction(db, models_validator, max_tokens, max_gen_tokens)
        self.duplicate_removal_validator = DuplicateRemoval()
        self.unsolvability_verification_validator = UnsolvabilityVerification(db, models_validator, self.sql_executability_validator, self.gt_satisfaction_validator)
        self.category_consistency_validator = CategoryConsistency(db, models_validator, categories)
        self.style_conformance_validator = StyleConformance(db, models_validator)
        self.feedback_quality_check_validator = FeedbackQualityCheck(db, models_validator)
        self.evidence_necessity_validator = EvidenceNecessity(db, models_validator)
        self.category_conformance_validator = CategoryConformance(db, models_validator, max_tokens, max_gen_tokens)

    def _save(self, questions: list[Question], label: str) -> None:
        if self.intermediate_results_folder is not None:
            save_questions(questions, self.intermediate_results_folder, label)

    def _save_tracking(self, stages: list[ValidationStageResult]) -> None:
        if self.intermediate_results_folder is not None:
            save_tracking(stages, self.intermediate_results_folder)

    def generate_for_model(self, model: Model, db_ids: list[str], categories: list[Category], styles: list[QuestionStyle], difficulties: list[QuestionDifficulty]) -> list[Question]:
        questions: list[Question] = []

        # --- Phase 1: Question-only generation (no SQL) ---
        prompts: list[str] = []
        constraints: list[type[BaseModel]] = []
        metadata: list[tuple[Category, str, QuestionStyle, QuestionDifficulty]] = []

        for category in categories:
            for db_id in db_ids:
                for style in styles:
                    for difficulty in difficulties:
                        prompt = get_generation_prompt(
                            db=self.db,
                            is_solvable=category.is_solvable(),
                            is_answerable=category.is_answerable(),
                            db_id=db_id,
                            name=category.get_name(),
                            definition=category.get_definition(),
                            examples=category.get_examples(),
                            output=category.get_output(),
                            question_style=style,
                            question_difficulty=difficulty
                        )
                        prompts.append(prompt)
                        constraints.append(category.get_output())
                        metadata.append((category, db_id, style, difficulty))

        model.init()
        responses: list[BaseModel | None] = model.generate_batch_with_constraints_unsafe(prompts, constraints)

        # Convert responses into Question instances (sql=None for answerable/ambiguous)
        assert len(responses) == len(prompts) * self.n_samples, "Number of responses does not match number of prompts times n_samples."
        for idx, response in enumerate(responses):
            if response is not None:
                category, db_id, style, difficulty = metadata[idx // self.n_samples]
                question = category.get_question(db_id, response, style, difficulty)
                questions.extend(question)

        # --- Phase 2: SQL generation for solvable questions ---
        sql_questions = [q for q in questions if q.category.is_solvable()]
        if sql_questions:
            sql_prompts = []
            for q in sql_questions:
                hidden_knowledge = q.hidden_knowledge if isinstance(q, QuestionUnanswerable) else None
                sql_prompts.append(get_sql_generation_prompt(
                    db=self.db,
                    db_id=q.db_id,
                    question=q.question,
                    evidence=q.evidence,
                    hidden_knowledge=hidden_knowledge,
                    question_difficulty=q.question_difficulty
                ))
            sql_constraints: list[type[BaseModel]] = [SQLGenerationOutput] * len(sql_prompts)

            # Override n=1 for Phase 2 (one SQL per question), then restore
            original_sampling_kwargs = model.sampling_kwargs
            model.sampling_kwargs = {**(model.sampling_kwargs or {}), 'n': 1}
            model.rebuild_sampling_params()

            sql_responses = model.generate_batch_with_constraints_unsafe(sql_prompts, sql_constraints)

            model.sampling_kwargs = original_sampling_kwargs

            assert len(sql_responses) == len(sql_prompts), \
                f"Phase 2: expected {len(sql_prompts)} SQL responses, got {len(sql_responses)}"

            # Assign SQL to solvable questions and filter out failures
            sql_fail_count = 0
            questions_with_sql: list[Question] = []
            sql_idx = 0
            for q in questions:
                if q.category.is_solvable():
                    resp = sql_responses[sql_idx]
                    sql_idx += 1
                    if resp is not None:
                        assert isinstance(resp, SQLGenerationOutput)
                        q.sql = resp.sql
                        questions_with_sql.append(q)
                    else:
                        sql_fail_count += 1
                else:
                    questions_with_sql.append(q)  # unanswerable, keep as-is

            if sql_fail_count > 0:
                logger.info("Phase 2: dropped %d/%d solvable questions (SQL generation failed)", sql_fail_count, len(sql_questions))

            questions = questions_with_sql

        model.close()

        # --- Difficulty re-assignment based on actual SQL ---
        for q in questions:
            if q.sql is not None:
                q.question_difficulty = classify_sql_difficulty(q.sql)

        return questions

    def generate(self, db_ids: list[str], categories: list[Category], styles: list[QuestionStyle], difficulties: list[QuestionDifficulty], resume: bool = False) -> list[Question]:
        all_questions: list[Question] = []

        for model in self.models:
            model_tag = os.path.basename(model.model_name)
            checkpoint_label = f"gen_{model_tag}"

            if resume:
                cached = load_checkpoint(self.intermediate_results_folder, checkpoint_label)
                if cached is not None:
                    all_questions.extend(cached)
                    continue

            questions = self.generate_for_model(model, db_ids, categories, styles, difficulties)
            self._save(questions, checkpoint_label)
            all_questions.extend(questions)

        return all_questions

    def apply_validator(self,
                        questions: list[Question],
                        validator: Validator,
                        intermediate_results_label: str,
                        check_if_amb_solvable: bool=False,
                        check_if_amb_unsolvable: bool=False,
                        check_if_answerable: bool=False,
                        check_if_answerable_with_evidence: bool=False) -> tuple[list[Question], ValidationStageResult | None]:
        questions_to_check: list[Question] = []
        # Determine which questions to validate based on their category properties and flags
        if check_if_amb_solvable:
            questions_to_check.extend([q for q in questions if not q.category.is_answerable() and q.category.is_solvable()])
        if check_if_amb_unsolvable:
            questions_to_check.extend([q for q in questions if not q.category.is_answerable() and not q.category.is_solvable()])
        if check_if_answerable:
            questions_to_check.extend([q for q in questions if q.category.is_answerable()])
        if check_if_answerable_with_evidence:
            questions_to_check.extend([q for q in questions if isinstance(q.category, AnswerableWithEvidenceCategory)])
        if not check_if_amb_solvable and not check_if_amb_unsolvable and not check_if_answerable and not check_if_answerable_with_evidence:
            questions_to_check = questions

        if len(questions_to_check) == 0:
            return questions, None  # Nothing to validate

        # Remove from the original list the questions that need to be validated
        # Use dict keyed by id() for O(1) lookup while preserving insertion order
        check_ids = {id(q): True for q in questions_to_check}
        questions = [q for q in questions if id(q) not in check_ids]

        result: ValidationStageResult = validator.validate(questions=questions_to_check)
        questions_to_check = [q for i, q in enumerate(questions_to_check) if result.validities[i]]

        # Combine validated questions with those that didn't need validation
        questions.extend(questions_to_check)

        self._save(questions, intermediate_results_label)
        return questions, result

    def validate(self, questions: list[Question], resume: bool = False) -> tuple[list[Question], GenerationTrackingReport]:
        # Ordered list of (validator, label, filter kwargs)
        validation_steps: list[tuple[Validator, str, dict]] = [
            (self.duplicate_removal_validator, "after_duplicate_removal", {}),
            (self.sql_executability_validator, "after_sql_executability_check", {"check_if_amb_solvable": True, "check_if_answerable": True}),
            (self.category_conformance_validator, "after_category_conformance_check", {}),
            (self.gt_satisfaction_validator, "after_gt_satisfaction_check", {"check_if_amb_solvable": True, "check_if_answerable": True}),
            (self.evidence_necessity_validator, "after_evidence_necessity_check", {"check_if_answerable_with_evidence": True}),
            (self.ambiguity_verification_validator, "after_ambiguity_verification", {"check_if_amb_solvable": True}),
            (self.unsolvability_verification_validator, "after_unsolvability_verification", {"check_if_amb_unsolvable": True}),
            (self.feedback_quality_check_validator, "after_feedback_quality_check", {"check_if_amb_unsolvable": True}),
            (self.category_consistency_validator, "after_category_consistency_check", {"check_if_amb_solvable": True, "check_if_amb_unsolvable": True}),
            (self.style_conformance_validator, "after_style_conformance", {}),
        ]

        tracking_stages: list[ValidationStageResult] = []

        # If resuming, find the latest completed validation checkpoint and skip to it
        resume_from_idx = -1
        if resume:
            saved_tracking = load_tracking(self.intermediate_results_folder)
            for i in range(len(validation_steps) - 1, -1, -1):
                _, label, _ = validation_steps[i]
                cached = load_checkpoint(self.intermediate_results_folder, label)
                if cached is not None:
                    questions = cached
                    resume_from_idx = i
                    tracking_stages = saved_tracking
                    logger.info("Resuming validation after '%s' (%d questions)", label, len(questions))
                    break

        if resume_from_idx < 0:
            self._save(questions, "initial")

        for i, (validator, label, kwargs) in enumerate(validation_steps):
            if i <= resume_from_idx:
                continue

            questions, result = self.apply_validator(questions, validator, label, **kwargs)
            if result is not None:
                tracking_stages.append(result)
            self._save_tracking(tracking_stages)

        report = GenerationTrackingReport(stages=tracking_stages)
        return questions, report
