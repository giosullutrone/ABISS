from dataset_dataclasses.question import QuestionUnanswerable
from validators.category_check import CategoryCheck
from validators.check_duplicate import CheckDuplicate
from generators.generator import Generator
from models.model import Model
from db_datasets.db_dataset import DBDataset


class GeneratorUnsolvable(Generator):
    def __init__(self, db: DBDataset, models: list[Model], models_validator: list[Model], n_samples: int, intermediate_results_folder: str | None) -> None:
        super().__init__(db, models, n_samples, solvable=False, intermediate_results_folder=intermediate_results_folder)
        self.db: DBDataset = db
        self.check_copy_validator = CheckDuplicate()
        self.category_check_validator = CategoryCheck(db, models_validator)

    def validate(self, questions: list[QuestionUnanswerable]) -> list[QuestionUnanswerable]:
        # Two validation steps will be performed:
        # 1. Check if the question is not a copy of another question in the dataset.
        # 2. Check if the question fit the category definition by majority voting among the models
        self.save_intermediate_results(questions, "initial")

        # Step 1: Check Copy Validation
        copy_valids: list[bool] = self.check_copy_validator.validate(questions=questions)
        questions = [q for i, q in enumerate(questions) if copy_valids[i]]
        self.save_intermediate_results(questions, "after_copy_check")

        # Step 2: Category Fit Validation
        category_valids: list[bool] = self.category_check_validator.validate(questions=questions)
        questions = [q for i, q in enumerate(questions) if category_valids[i]]
        self.save_intermediate_results(questions, "after_category_check")
        return questions