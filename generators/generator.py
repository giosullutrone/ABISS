from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty
from categories.category import Category
from db_datasets.db_dataset import DBDataset
from models.model import Model
import json
import os
from generators.generator_prompt import get_generation_prompt
from pydantic import BaseModel
from validators.sql_executable import SQLExecutable
from validators.check_ambiguousness import CheckAmbiguousness
from validators.check_gt import CheckGT
from validators.check_duplicate import CheckDuplicate
from validators.check_unsolvable import CheckUnsolvable
from validators.category_comparison import CategoryComparison
from validators.style_difficulty_check import StyleDifficultyCheck
from validators.feedback_quality_check import FeedbackQualityCheck
from validators.validator import Validator


class Generator:
    def __init__(self, 
                 db: DBDataset, 
                 models: list[Model], 
                 models_validator: list[Model],
                 categories: list[Category],
                 n_samples: int, 
                 intermediate_results_folder: str | None) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.n_samples: int = n_samples
        self.intermediate_results_folder = intermediate_results_folder

        self.sql_validator = SQLExecutable(db)
        self.check_ambiguousness_validator = CheckAmbiguousness(db, models_validator)
        self.check_gt_validator = CheckGT(db, models_validator)
        self.check_copy_validator = CheckDuplicate()
        self.check_unsolvable_validator = CheckUnsolvable(db, models_validator, self.sql_validator, self.check_gt_validator)
        self.category_comparison_validator = CategoryComparison(db, models_validator, categories)
        self.style_difficulty_check_validator = StyleDifficultyCheck(db, models_validator)
        self.feedback_quality_check_validator = FeedbackQualityCheck(db, models_validator)

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
        responses: list[BaseModel] = model.generate_batch_with_constraints(prompts, constraints)
        model.close()

        # Convert the responses into Question instances
        assert len(responses) == len(prompts) * self.n_samples, "Number of responses does not match number of prompts times n_samples."
        for idx, response in enumerate(responses):
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
                        check_if_answerable: bool=False) -> list[Question]:
        questions_to_check: list[Question] = []
        # Determine which questions to validate based on their category properties and flags
        if check_if_amb_solvable:
            questions_to_check.extend([q for q in questions if not q.category.is_answerable() and q.category.is_solvable()])
        if check_if_amb_unsolvable:
            questions_to_check.extend([q for q in questions if not q.category.is_answerable() and not q.category.is_solvable()])
        if check_if_answerable:
            questions_to_check.extend([q for q in questions if q.category.is_answerable()])
        if not check_if_amb_solvable and not check_if_amb_unsolvable and not check_if_answerable:
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

        # Step 1: Check Copy Validation (remove duplicates)
        questions = self.apply_validator(questions, self.check_copy_validator, "after_copy_check")

        # Step 2: SQL Executability Validation if answerable or amb solvable
        # Must check SQL is valid before checking if it satisfies requirements
        questions = self.apply_validator(questions, self.sql_validator, "after_sql_executability_check", check_if_amb_solvable=True, check_if_answerable=True)

        # Step 3: GT Satisfaction Validation if answerable or amb solvable
        # Check that SQL actually answers the question correctly before checking other properties
        questions = self.apply_validator(questions, self.check_gt_validator, "after_gt_satisfaction_check", check_if_amb_solvable=True, check_if_answerable=True)

        # Step 4: Ambiguity Validation if amb solvable
        # After confirming SQL is valid and correct, check if question is actually ambiguous
        questions = self.apply_validator(questions, self.check_ambiguousness_validator, "after_ambiguity_check", check_if_amb_solvable=True)

        # Step 5: Unsolvability Validation if amb unsolvable
        # Check if unsolvable questions are truly unsolvable
        questions = self.apply_validator(questions, self.check_unsolvable_validator, "after_unsolvability_check", check_if_amb_unsolvable=True)
        
        # Step 6: Feedback Quality Check if amb unsolvable
        # After confirming unsolvability, check that feedback correctly explains why
        questions = self.apply_validator(questions, self.feedback_quality_check_validator, "after_feedback_quality_check", check_if_amb_unsolvable=True)

        # Step 7: Other Categories Check Validation
        # Verify question doesn't fit better in a different category
        questions = self.apply_validator(questions, self.category_comparison_validator, "after_category_comparison_check")

        # Step 8: Check style and difficulty
        # Final check that question matches intended style and difficulty
        questions = self.apply_validator(questions, self.style_difficulty_check_validator, "after_style_difficulty_check")
        return questions
