from validators.validator import Validator
from categories.category import Category
from dataset_dataclasses.question import QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from tqdm import tqdm


class SQLExecutable(Validator):
    def __init__(self, db: DBDataset) -> None:
        self.db: DBDataset = db

    def validate(self, questions: list[QuestionUnanswerable]) -> list[bool]:
        def check_and_validate(question: QuestionUnanswerable) -> bool:
            assert question.sql is not None, "GT SQL query is None."
            return self.db.query_has_results(
                db_id=question.db_id,
                sql_query=question.sql,
                db_sql_manipulation=None
            )
        return [check_and_validate(question) for question in tqdm(questions, desc="SQL Executability Validation")]
