from validators.validator import Validator
from dataset_dataclasses.question import Question
from db_datasets.db_dataset import DBDataset
from tqdm import tqdm


class SQLExecutable(Validator):
    def __init__(self, db: DBDataset) -> None:
        self.db: DBDataset = db

    def validate(self, questions: list[Question]) -> list[bool]:
        def check_and_validate(question: Question) -> bool:
            assert question.sql is not None, "GT SQL query is None."
            return self.db.query_has_results(
                db_id=question.db_id,
                sql_query=question.sql
            )
        return [check_and_validate(question) for question in tqdm(questions, desc="SQL Executability Validation")]
