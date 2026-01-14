from .bird_utils import generate_schema_prompt, execute_sql_query, generate_combined_prompts_one, extract_last_sql_query_from_block, get_table_columns, get_table_id_columns
import os
import json
from dataset_dataclasses.question import Question


class DBDataset:
    def __init__(self, db_root_path: str) -> None:
        self.db_root_path: str = db_root_path
        self.db_name = "BIRD"

    def get_db_path(self, db_id: str) -> str:
        return os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")

    def get_questions(self, question_path: str) -> list[Question]:
        questions_json = json.load(open(question_path, 'r'))
        questions = []
        for question in questions_json:
            questions.append(
                Question(
                    db_id=question['db_id'],
                    question=question['question'],
                    evidence=question['evidence'],
                    sql=question['SQL']
                )
            )
        return questions

    def get_table_columns(self, db_id: str, table_name: str, db_sql_manipulation: str | None) -> list[str] | None:
        return get_table_columns(
            db_path=self.get_db_path(db_id),
            table_name=table_name,
            db_sql_manipulation=db_sql_manipulation
        )
    
    def get_table_id_columns(self, db_id: str, table_name: str, db_sql_manipulation: str | None) -> list[str] | None:
        return get_table_id_columns(
            db_path=self.get_db_path(db_id),
            table_name=table_name,
            db_sql_manipulation=db_sql_manipulation
        )

    def get_schema_prompt(self, db_id: str, rows: int | None, db_sql_manipulation: str | None) -> str:
        return generate_schema_prompt(
            db_path=self.get_db_path(db_id),
            num_rows=rows,
            db_sql_manipulation=db_sql_manipulation
        )

    def get_prompt(self, db_id: str, question: str, evidence: str | None, num_rows: int | None, db_sql_manipulation: str | None) -> str:
        return generate_combined_prompts_one(
            db_path=self.get_db_path(db_id),
            question=question,
            knowledge=evidence,
            num_rows=num_rows,
            db_sql_manipulation=db_sql_manipulation
        )

    def extract_last_sql_query_from_block(self, text: str) -> str | None:
        return extract_last_sql_query_from_block(text)

    def execute_query(self, db_id: str, sql_query: str, db_sql_manipulation: str | None) -> list[tuple] | None:
        try:
            db_path = self.get_db_path(db_id)
            return execute_sql_query(db_path=db_path, sql_query=sql_query, db_sql_manipulation=db_sql_manipulation)
        except Exception:
            return None

    def validate_query(self, db_id: str, sql_query: str, db_sql_manipulation: str | None) -> bool:
        """
        Check if the SQL query is valid for the given database.
        """
        results = self.execute_query(db_id=db_id, sql_query=sql_query, db_sql_manipulation=db_sql_manipulation)
        return results is not None
        
    def query_has_results(self, db_id: str, sql_query: str, db_sql_manipulation: str | None) -> bool:
        results = self.execute_query(db_id=db_id, sql_query=sql_query, db_sql_manipulation=db_sql_manipulation)
        return results is not None and len(results) > 0

    def compare_query_results(self, 
                              db_id: str, 
                              sql_query_1: str, 
                              sql_query_2: str, 
                              db_sql_manipulation: str | None) -> bool | None:
        result_1 = self.execute_query(db_id=db_id, sql_query=sql_query_1, db_sql_manipulation=db_sql_manipulation)
        result_2 = self.execute_query(db_id=db_id, sql_query=sql_query_2, db_sql_manipulation=db_sql_manipulation)
        if result_1 is None or result_2 is None:
            return None
        return result_1 == result_2
    
    def get_db_ids(self) -> list[str]:
        return [name for name in os.listdir(self.db_root_path) if os.path.isdir(os.path.join(self.db_root_path, name))]