import os
from db_datasets.prompts import generate_combined_prompts_one, generate_schema_prompt, extract_last_sql_query_from_block
from typing import Any
import sqlite3
import time


class DBDataset:
    def __init__(self, db_root_path: str, db_name: str) -> None:
        self.db_root_path: str = db_root_path
        self.db_name = db_name

    ### Database interaction methods ###
    def _get_db_path(self, db_id: str) -> str:
        return os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")

    def get_db_ids(self) -> list[str]:
        return [name for name in os.listdir(self.db_root_path) if os.path.isdir(os.path.join(self.db_root_path, name))]

    ### Prompt generation and query execution methods ###
    def get_schema_prompt(self, db_id: str, rows: int | None) -> str:
        return generate_schema_prompt(
            db_path=self._get_db_path(db_id),
            num_rows=rows
        )

    def get_prompt(self, db_id: str, question: str, evidence: str | None, num_rows: int | None) -> str:        
        return generate_combined_prompts_one(
            db_path=self._get_db_path(db_id),
            question=question,
            knowledge=evidence,
            num_rows=num_rows
        )

    def extract_last_sql_query_from_block(self, text: str) -> str | None:
        return extract_last_sql_query_from_block(text)

    ### Query execution and result comparison methods ###
    def _execute_query(
        self,
        db_id: str,
        sql_query: str,
        max_seconds: float | None = 30.0,
    ) -> list[tuple[Any, ...]]:
        """
        Execute the given SQL query on the specified database and return the results.
        If max_seconds is provided, the query will be aborted if it runs longer than that.
        """
        # Open in immutable mode to avoid locking issues on network filesystems
        db_path = self._get_db_path(db_id)
        conn = sqlite3.connect(f'file:{db_path}?immutable=1', uri=True)
        cursor = conn.cursor()

        # Set up timeout via progress handler
        if max_seconds is not None:
            start = time.monotonic()

            def progress_handler() -> int:
                # Called every N VM steps; return non-zero to abort
                if time.monotonic() - start > max_seconds:
                    return 1  # abort query -> raises sqlite3.OperationalError
                return 0

            # Call handler every 1000 "virtual machine" steps (tune as needed)
            conn.set_progress_handler(progress_handler, 1000)

        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
        finally:
            # Always clear handler and close connection
            conn.set_progress_handler(None, 0)
            conn.close()
        return results

    def execute_query(self, db_id: str, sql_query: str) -> list[tuple] | None:
        try:
            return self._execute_query(db_id=db_id, sql_query=sql_query)
        except Exception:
            return None
    
    def query_has_results(self, db_id: str, sql_query: str) -> bool:
        results = self.execute_query(db_id=db_id, sql_query=sql_query)
        return results is not None and len(results) > 0

    def compare_query_results(self, 
                              db_id: str, 
                              sql_query_1: str, 
                              sql_query_2: str) -> bool | None:
        result_1 = self.execute_query(db_id=db_id, sql_query=sql_query_1)
        result_2 = self.execute_query(db_id=db_id, sql_query=sql_query_2)
        if result_1 is None or result_2 is None:
            return None
        return result_1 == result_2
