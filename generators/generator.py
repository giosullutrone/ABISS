from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty
from categories.category import Category
from db_datasets.db_dataset import DBDataset
from models.model import Model
import json
import os
from generators.prompts.generator_prompt import get_generation_prompt
from pydantic import BaseModel
from validators.sql_executability import SQLExecutability
from validators.ambiguity_verification import AmbiguityVerification
from validators.gt_satisfaction import GTSatisfaction
from validators.duplicate_removal import DuplicateRemoval
from validators.unsolvability_verification import UnsolvabilityVerification
from validators.category_consistency import CategoryConsistency
from validators.style_conformance import StyleConformance
from validators.difficulty_conformance import DifficultyConformance
from validators.feedback_quality_check import FeedbackQualityCheck
from validators.evidence_necessity import EvidenceNecessity
from categories.answerable_with_evidence import AnswerableWithEvidenceCategory
from validators.validator import Validator


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
        self.difficulty_conformance_validator = DifficultyConformance()
        self.feedback_quality_check_validator = FeedbackQualityCheck(db, models_validator)
        self.evidence_necessity_validator = EvidenceNecessity(db, models_validator)

    def save_intermediate_results(self, questions: list[Question], stage: str) -> None:
        if self.intermediate_results_folder is not None:
            os.makedirs(self.intermediate_results_folder, exist_ok=True)
            with open(os.path.join(self.intermediate_results_folder, f"intermediate_{stage}.json"), "w", encoding="utf-8") as f:
                json.dump([q.to_dict() for q in questions], f, ensure_ascii=False, indent=4)

    def generate_for_model(self, model: Model, db_ids: list[str], categories: list[Category], styles: list[QuestionStyle], difficulties: list[QuestionDifficulty]) -> list[Question]:
        questions: list[Question] = []
        
        # Generate all prompts and constraints for all categories at once
        prompts: list[str] = []
        constraints: list[type[BaseModel]] = []
        metadata: list[tuple[Category, str, QuestionStyle, QuestionDifficulty]] = []  # Track (category, db_id, style, difficulty) for each prompt
        
        for category in categories:
            for db_id in db_ids:
                # Generate prompts for all combinations of style and difficulty
                for style in styles:
                    for difficulty in difficulties:
                        # Prepare the generation prompt
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
        
        # Use the model to generate all questions in a single call
        model.init()
        responses: list[BaseModel | None] = model.generate_batch_with_constraints_unsafe(prompts, constraints)
        model.close()

        # Convert the responses into Question instances
        assert len(responses) == len(prompts) * self.n_samples, "Number of responses does not match number of prompts times n_samples."
        for idx, response in enumerate(responses):
            if response is not None:
                category, db_id, style, difficulty = metadata[idx // self.n_samples]
                question = category.get_question(db_id, response, style, difficulty)
                questions.extend(question)
        return questions

    def generate(self, db_ids: list[str], categories: list[Category], styles: list[QuestionStyle], difficulties: list[QuestionDifficulty]) -> list[Question]:
        all_questions: list[Question] = []
        for model in self.models:
            questions = self.generate_for_model(model, db_ids, categories, styles, difficulties)
            all_questions.extend(questions)
        return all_questions
    
    def apply_validator(self, 
                        questions: list[Question], 
                        validator: Validator, 
                        intermediate_results_label: str,
                        check_if_amb_solvable: bool=False,
                        check_if_amb_unsolvable: bool=False,
                        check_if_answerable: bool=False,
                        check_if_answerable_with_evidence: bool=False) -> list[Question]:
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
            return questions  # Nothing to validate

        # Remove from the original list the questions that need to be validated
        questions = [q for q in questions if q not in questions_to_check]
        
        validities: list[bool] = validator.validate(questions=questions_to_check)
        questions_to_check = [q for i, q in enumerate(questions_to_check) if validities[i]]

        # Combine validated questions with those that didn't need validation
        questions.extend(questions_to_check)

        self.save_intermediate_results(questions, intermediate_results_label)
        return questions

    def validate(self, questions: list[Question]) -> list[Question]:
        self.save_intermediate_results(questions, "initial")

        # Step 1: Duplicate Removal (remove duplicates)
        questions = self.apply_validator(questions, self.duplicate_removal_validator, "after_duplicate_removal")

        # Step 2: SQL Executability Validation if answerable or amb solvable
        # Must check SQL is valid before checking if it satisfies requirements
        questions = self.apply_validator(questions, self.sql_executability_validator, "after_sql_executability_check", check_if_amb_solvable=True, check_if_answerable=True)

        # Step 3: GT Satisfaction Validation if answerable or amb solvable
        # Check that SQL actually answers the question correctly before checking other properties
        questions = self.apply_validator(questions, self.gt_satisfaction_validator, "after_gt_satisfaction_check", check_if_amb_solvable=True, check_if_answerable=True)

        # Step 4: Evidence Necessity Validation if answerable with evidence
        # Verify that evidence is truly needed — models should NOT be able to produce equivalent SQL without it
        questions = self.apply_validator(questions, self.evidence_necessity_validator, "after_evidence_necessity_check", check_if_answerable_with_evidence=True)

        # Step 5: Ambiguity Verification if amb solvable
        # After confirming SQL is valid and correct, check if question is actually ambiguous
        questions = self.apply_validator(questions, self.ambiguity_verification_validator, "after_ambiguity_verification", check_if_amb_solvable=True)

        # Step 6: Unsolvability Verification if amb unsolvable
        # Check if unsolvable questions are truly unsolvable
        questions = self.apply_validator(questions, self.unsolvability_verification_validator, "after_unsolvability_verification", check_if_amb_unsolvable=True)
        
        # Step 7: Feedback Quality Check if amb unsolvable
        # After confirming unsolvability, check that feedback correctly explains why
        questions = self.apply_validator(questions, self.feedback_quality_check_validator, "after_feedback_quality_check", check_if_amb_unsolvable=True)

        # Step 8: Category Consistency Check Validation
        # Verify question doesn't fit better in a different category
        questions = self.apply_validator(questions, self.category_consistency_validator, "after_category_consistency_check", check_if_amb_solvable=True, check_if_amb_unsolvable=True)

        # Step 9: Difficulty Conformance check (automated keyword-based SQL analysis)
        # Verify SQL complexity matches the specified difficulty level
        questions = self.apply_validator(questions, self.difficulty_conformance_validator, "after_difficulty_conformance")

        # Step 10: Style Conformance check (LLM-based)
        # Verify question matches the intended style
        questions = self.apply_validator(questions, self.style_conformance_validator, "after_style_conformance")
        return questions
