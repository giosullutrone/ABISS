from generators.generator_solvable import GeneratorSolvable
from generators.generator_unsolvable import GeneratorUnsolvable
from generators.generator_answerable import GeneratorAnswerable
from dataset_dataclasses.question import Question
from models.model import Model
from categories.category import Category
from db_datasets.db_dataset import DBDataset


class Chain:
    def __init__(self, 
                 models: list[Model], 
                 generator_answerable: GeneratorAnswerable,
                 generator_solvable: GeneratorSolvable, 
                 generator_unsolvable: GeneratorUnsolvable, 
                 categories: list[Category]) -> None:
        self.models: list[Model] = models
        self.generator_answerable: GeneratorAnswerable = generator_answerable
        self.generator_solvable: GeneratorSolvable = generator_solvable
        self.generator_unsolvable: GeneratorUnsolvable = generator_unsolvable
        self.categories: list[Category] = categories

    def generate(self, db: DBDataset) -> list[Question]:
        db_ids: list[str] = db.get_db_ids()

        questions_unanswerable: list[Question] = []

        categories_answerable = [category for category in self.categories if category.is_answerable()]
        if categories_answerable:
            generated_questions = self.generator_answerable.generate(db_ids, categories_answerable)
            valid_questions = self.generator_answerable.validate(generated_questions)

        categories_solvable = [category for category in self.categories if category.is_solvable() and not category.is_answerable()]
        if categories_solvable:
            generated_questions = self.generator_solvable.generate(db_ids, categories_solvable)
            valid_questions = self.generator_solvable.validate(generated_questions)
            questions_unanswerable.extend([q for q, valid in zip(generated_questions, valid_questions) if valid])

        categories_unsolvable = [category for category in self.categories if not category.is_solvable() and not category.is_answerable()]
        if categories_unsolvable:
            generated_questions = self.generator_unsolvable.generate(db_ids, categories_unsolvable)
            valid_questions = self.generator_unsolvable.validate(generated_questions)
            questions_unanswerable.extend([q for q, valid in zip(generated_questions, valid_questions) if valid])
        return questions_unanswerable
