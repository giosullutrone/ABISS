from dataset_dataclasses.question import Question, QuestionDifficulty, QuestionStyle
from models.model import Model
from categories.category import Category
from db_datasets.db_dataset import DBDataset
from generators.generator import Generator


class Chain:
    def __init__(self, 
                 models: list[Model], 
                 generator: Generator,
                 categories: list[Category],
                 styles: list[QuestionStyle],
                 difficulties: list[QuestionDifficulty],
                 db_ids: list[str]) -> None:
        self.models: list[Model] = models
        self.generator: Generator = generator
        self.categories: list[Category] = categories
        self.styles: list[QuestionStyle] = styles
        self.difficulties: list[QuestionDifficulty] = difficulties
        self.db_ids: list[str] = db_ids

    def generate(self) -> list[Question]:
        # Generate questions
        generated_questions = self.generator.generate(self.db_ids, self.categories, self.styles, self.difficulties)
        # Validate questions
        questions = self.generator.validate(generated_questions)
        return questions