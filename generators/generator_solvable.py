from generators.generator import Generator
from categories.category import Category
from models.model import Model
from dataset_dataclasses.question import QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from validators.sql_executable import SQLExecutable
from validators.category_check import CategoryCheck
from validators.check_ambiguousness import CheckAmbiguousness
from validators.check_gt import CheckGT
from validators.check_duplicate import CheckDuplicate


class GeneratorSolvable(Generator):
    def __init__(self, db: DBDataset, models: list[Model], models_validator: list[Model], n_samples: int, intermediate_results_folder: str | None) -> None:
        super().__init__(db, models, n_samples, solvable=True, intermediate_results_folder=intermediate_results_folder)
        self.db: DBDataset = db
        self.sql_validator = SQLExecutable(db)
        self.category_check_validator = CategoryCheck(db, models_validator)
        self.check_ambiguousness_validator = CheckAmbiguousness(db, models_validator)
        self.check_gt_validator = CheckGT(db, models_validator)
        self.check_copy_validator = CheckDuplicate()

    def validate(self, questions: list[QuestionUnanswerable]) -> list[QuestionUnanswerable]:
        # Five validation steps will be performed:
        # 1. Check if the question is not a copy of another question in the dataset.
        # 2. Check if the generated GT SQL executes without errors on the database schema.
        # 3. Check if the question fit the category definition by majority voting among the models.
        # 4. Check if the question is indeed ambiguous and cannot be answered by any of the models without the hidden knowledge.
        # 5. Check if the model, given also the results of the GT SQL query, deems the result satisfactory.
        self.save_intermediate_results(questions, "initial")

        # Step 1: Check Copy Validation
        copy_valids: list[bool] = self.check_copy_validator.validate(questions=questions)
        questions = [q for i, q in enumerate(questions) if copy_valids[i]]
        self.save_intermediate_results(questions, "after_copy_check")

        # Step 2: SQL Executability Validation
        sql_valids: list[bool] = self.sql_validator.validate(questions=questions)
        questions = [q for i, q in enumerate(questions) if sql_valids[i]]
        self.save_intermediate_results(questions, "after_sql_executability_check")

        # Step 3: Category Fit Validation
        category_valids: list[bool] = self.category_check_validator.validate(questions=questions)
        questions = [q for i, q in enumerate(questions) if category_valids[i]]
        self.save_intermediate_results(questions, "after_category_check")

        # Step 4: Ambiguity Validation
        ambiguity_valids: list[bool] = self.check_ambiguousness_validator.validate(questions=questions)
        questions = [q for i, q in enumerate(questions) if ambiguity_valids[i]]
        self.save_intermediate_results(questions, "after_ambiguity_check")

        # Step 5: GT Satisfaction Validation
        gt_satisfaction_valids: list[bool] = self.check_gt_validator.validate(questions=questions)
        questions = [q for i, q in enumerate(questions) if gt_satisfaction_valids[i]]
        self.save_intermediate_results(questions, "after_gt_satisfaction_check")
        return questions
