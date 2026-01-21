from validators.validator import Validator
from validators.sql_executable import SQLExecutable
from validators.check_gt import CheckGT
from dataset_dataclasses.question import Question
from models.model import Model
from db_datasets.db_dataset import DBDataset
from copy import deepcopy


class CheckUnsolvable(Validator):
    """
    Validator for unsolvable questions (questions that shouldn't have an SQL answer).
    
    The validation process:
    1. Generate SQL attempts for unsolvable questions
    2. Apply generated SQLs to copies of the questions
    3. Check if SQLs are executable
       - If not executable -> Valid (confirmed unsolvable)
       - If executable -> Check if SQL answers the question correctly
         - If answers correctly -> Invalid (question is actually solvable)
         - If doesn't answer correctly -> Valid (confirmed unsolvable)
    """
    
    def __init__(self, db: DBDataset, models: list[Model], sql_executable_validator: SQLExecutable, check_gt_validator: CheckGT) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.sql_executable_validator = sql_executable_validator
        self.check_gt_validator = check_gt_validator

    def validate(self, questions: list[Question]) -> list[bool]:
        valids: list[bool] = [True for _ in questions]

        # Generate SQLs for each question using all models
        sqls: list[list[str | None]] = [self.db.generate_sqls_unsafe(model, questions) for model in self.models]

        # Create all question copies with generated SQLs at once
        all_questions_with_generated_sql: list[Question] = []
        question_mapping: list[int] = []  # Maps each generated question back to its original index
        
        for i, question in enumerate(questions):
            for model_sqls in sqls:
                generated_sql = model_sqls[i]
                if generated_sql is not None:  # Only validate questions with generated SQL
                    question_copy = deepcopy(question)
                    question_copy.sql = generated_sql
                    all_questions_with_generated_sql.append(question_copy)
                    question_mapping.append(i)
        
        if not all_questions_with_generated_sql:
            # No SQL was generated for any question, all remain valid (confirmed unsolvable)
            return valids
        
        # Check which generated SQLs are executable
        executable_flags = self.sql_executable_validator.validate(all_questions_with_generated_sql)
        executable_questions = [q for j, q in enumerate(all_questions_with_generated_sql) if executable_flags[j]]
        executable_mapping = [question_mapping[j] for j, flag in enumerate(executable_flags) if flag]
        
        if not executable_questions:
            # None of the generated SQLs are executable, all questions remain valid (confirmed unsolvable)
            return valids
        
        # For executable SQLs, check if they correctly answer the question
        gt_validation_flags = self.check_gt_validator.validate(executable_questions)
        
        # Mark questions as invalid if any of their generated SQLs are executable AND correctly answer the question
        for j, is_valid_gt in enumerate(gt_validation_flags):
            if is_valid_gt:
                original_idx = executable_mapping[j]
                valids[original_idx] = False
        return valids
