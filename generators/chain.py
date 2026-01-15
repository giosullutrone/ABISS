from dataset_dataclasses.question import Question
from models.model import Model
from categories.category import Category
from db_datasets.db_dataset import DBDataset
from generators.generator import Generator


class Chain:
    def __init__(self, 
                 models: list[Model], 
                 generator: Generator,
                 categories: list[Category]) -> None:
        self.models: list[Model] = models
        self.generator: Generator = generator
        self.categories: list[Category] = categories

    def generate(self, db: DBDataset) -> list[Question]:
        db_ids: list[str] = db.get_db_ids()
        # Generate questions
        generated_questions = self.generator.generate(db_ids, self.categories)
        # Validate questions
        questions = self.generator.validate(generated_questions)
        return questions