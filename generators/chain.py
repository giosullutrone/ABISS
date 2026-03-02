from dataset_dataclasses.question import Question, QuestionDifficulty, QuestionStyle
from dataset_dataclasses.council_tracking import GenerationTrackingReport
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

    def generate(self) -> tuple[list[Question], GenerationTrackingReport]:
        # Skip generation entirely if a checkpoint exists
        checkpoint = self.generator.try_load_checkpoint("after_sql_executability_check")
        if checkpoint is not None:
            questions, report = self.generator.validate(checkpoint, skip_through_sql_executability=True)
        else:
            generated_questions = self.generator.generate(self.db_ids, self.categories, self.styles, self.difficulties)
            questions, report = self.generator.validate(generated_questions)
        return questions, report