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

        questions: list[Question] = []

        categories_answerable = [category for category in self.categories if category.is_answerable()]
        categories_solvable = [category for category in self.categories if category.is_solvable() and not category.is_answerable()]
        categories_unsolvable = [category for category in self.categories if not category.is_solvable() and not category.is_answerable()]

        # All generators have the same generation function from the parent class so we can call it once on all categories at once
        generated_questions = self.generator_answerable.generate(db_ids, self.categories)

        # Now we need to validate and filter the questions according to their category type
        for category in categories_answerable:
            questions_category = [q for q in generated_questions if q.category == category]
            validated_questions = self.generator_answerable.validate(questions_category)
            questions.extend(validated_questions)

        for category in categories_solvable:
            questions_category = [q for q in generated_questions if q.category == category]
            validated_questions = self.generator_solvable.validate(questions_category)
            questions.extend(validated_questions)

        for category in categories_unsolvable:
            questions_category = [q for q in generated_questions if q.category == category]
            validated_questions = self.generator_unsolvable.validate(questions_category)
            questions.extend(validated_questions)
        return questions