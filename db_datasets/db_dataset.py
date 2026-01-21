import os
from typing import Any
import sqlite3
import time
from db_datasets.sql_generation_prompts import SQLGenerationResponse, get_sql_generation_prompt, get_sql_result
from db_datasets.sql_schema_prompts import generate_schema_prompt
from dataset_dataclasses.question import Question
from models.model import Model


class DBDataset:
    def __init__(self, db_root_path: str, db_name: str) -> None:
        self.db_root_path: str = db_root_path
        self.db_name = db_name

    ### Database interaction methods ###
    def _get_db_path(self, db_id: str) -> str:
        return os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")

    def get_db_ids(self) -> list[str]:
        return [name for name in os.listdir(self.db_root_path) if os.path.isdir(os.path.join(self.db_root_path, name))]

    ### Generation methods ###
    def get_schema_prompt(self, db_id: str, rows: int | None) -> str:
        return generate_schema_prompt(
            db_path=self._get_db_path(db_id),
            num_rows=rows
        )

    def generate_sqls(self, model: Model, questions: list[Question]) -> list[str]:
        # Generate the SQL generation prompts
        prompts = [get_sql_generation_prompt(
            db=self,
            db_id=q.db_id,
            question=q.question,
            evidence=q.evidence
        ) for q in questions]

        model.init()
        responses = model.generate_batch_with_constraints(prompts, [SQLGenerationResponse for _ in prompts])
        model.close()
        return [get_sql_result(response) for response in responses]
    
    def generate_sqls_unsafe(self, model: Model, questions: list[Question]) -> list[str | None]:
        # Generate the SQL generation prompts
        prompts = [get_sql_generation_prompt(
            db=self,
            db_id=q.db_id,
            question=q.question,
            evidence=q.evidence
        ) for q in questions]

        model.init()
        responses = model.generate_batch_with_constraints_unsafe(prompts, [SQLGenerationResponse for _ in prompts])
        model.close()
        return [get_sql_result(response) if response is not None else None for response in responses]
    
    ### Query execution and result comparison methods ###
    def _execute_query(
        self,
        db_id: str,
        sql_query: str,
        max_seconds: float | None = 30.0,
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        """
        Execute the given SQL query on the specified database and return column names and results.
        If max_seconds is provided, the query will be aborted if it runs longer than that.
        
        Returns:
            Tuple of (column_names, results)
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
            # Get column names from cursor description
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
        finally:
            # Always clear handler and close connection
            conn.set_progress_handler(None, 0)
            conn.close()
        return column_names, results

    def execute_query(self, db_id: str, sql_query: str) -> list[tuple] | None:
        """Execute query and return only results (for backward compatibility)."""
        try:
            _, results = self._execute_query(db_id=db_id, sql_query=sql_query)
            return results
        except Exception:
            return None
    
    def execute_query_with_columns(self, db_id: str, sql_query: str) -> tuple[list[str], list[tuple]] | None:
        """Execute query and return both column names and results."""
        try:
            return self._execute_query(db_id=db_id, sql_query=sql_query)
        except Exception:
            return None
    
    def query_has_results(self, db_id: str, sql_query: str) -> bool:
        results = self.execute_query(db_id=db_id, sql_query=sql_query)
        return results is not None and len(results) > 0

    def _compare_results_relaxed(self, 
                                  cols_generated: list[str],
                                  result_generated: list[tuple[Any, ...]], 
                                  cols_ground_truth: list[str],
                                  result_ground_truth: list[tuple[Any, ...]]) -> bool:
        """
        Compare query results using relaxed semantic equivalence criteria from Floratou et al.
        
        Rules:
        1. Row ordering is ignored
        2. Generated query can return a superset of columns (column names must match)
        3. Columns can be in any order, we match by name and reorder accordingly
        
        Returns:
            True if results are semantically equivalent under relaxed criteria
        """
        # Handle empty results
        if len(result_generated) != len(result_ground_truth):
            return False
        
        if len(result_generated) == 0:
            return True
        
        # Check that all ground truth columns are present in generated columns
        cols_gt_set = set(cols_ground_truth)
        cols_gen_set = set(cols_generated)
        
        if not cols_gt_set.issubset(cols_gen_set):
            return False
        
        # Build a mapping from ground truth column positions to generated column positions
        column_mapping = [cols_generated.index(col) for col in cols_ground_truth]
        
        # Reorder generated results to match ground truth column order
        projected_generated = [
            tuple(row[i] for i in column_mapping) 
            for row in result_generated
        ]
        
        # Compare as unordered sets (ignoring row ordering)
        set_generated = set(projected_generated)
        set_ground_truth = set(result_ground_truth)
        
        return set_generated == set_ground_truth

    def compare_query_results(self, 
                              db_id: str, 
                              sql_query_1: str, 
                              sql_query_2: str) -> bool | None:
        """
        Compare results of two SQL queries using relaxed semantic equivalence.
        
        sql_query_1 is treated as the generated query (can have more columns)
        sql_query_2 is treated as the ground truth query
        
        Returns:
            True if semantically equivalent, False if not, None if execution error
        """
        result_1 = self.execute_query_with_columns(db_id=db_id, sql_query=sql_query_1)
        result_2 = self.execute_query_with_columns(db_id=db_id, sql_query=sql_query_2)
        if result_1 is None or result_2 is None:
            return None
        
        cols_1, data_1 = result_1
        cols_2, data_2 = result_2
        
        return self._compare_results_relaxed(cols_1, data_1, cols_2, data_2)
