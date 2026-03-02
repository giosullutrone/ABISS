from validators.validator import Validator
from dataset_dataclasses.question import Question, QuestionUnanswerable
from dataset_dataclasses.council_tracking import ValidationStageResult
from tqdm import tqdm
import re


def mask_sql_values(sql: str) -> str:
    """
    Extract SQL template by masking values in the SQL query.
    Numbers, strings, and other literal values are replaced with [MASK].
    
    Example:
        "SELECT name FROM school WHERE age > 18" -> "SELECT name FROM school WHERE age > [MASK]"
        "SELECT * FROM users WHERE name = 'John'" -> "SELECT * FROM users WHERE name = [MASK]"
    """    
    # Mask single-quoted string literals only.
    # Double quotes in SQLite are identifier quotes (like backticks in MySQL),
    # not string delimiters, so we must NOT mask them.
    masked_sql = re.sub(r"'(?:[^']|'')*'", "[MASK]", sql)
    
    # Mask numeric literals (integers and floats)
    # Match standalone numbers (not part of identifiers)
    masked_sql = re.sub(r'\b\d+\.?\d*\b', "[MASK]", masked_sql)
    return masked_sql


class DuplicateRemoval(Validator):
    def validate(self, questions: list[Question]) -> ValidationStageResult:
        # Check if any questions is a copy of another question in the dataset.
        # Deduplication based on:
        # 1. Question text + hidden knowledge (if applicable)
        # 2. SQL template (masked values as per OMNI-SQL paper)
        seen_questions: set[tuple[str, str | None]] = set()
        seen_sql_templates: set[tuple[str, str | None]] = set()
        valids: list[bool] = []
        
        for question in tqdm(questions, desc="Check Duplicate Validation"):
            # Check for duplicate question + hidden knowledge (if applicable)
            question_key = (question.question, question.hidden_knowledge if isinstance(question, QuestionUnanswerable) else None)
            is_in_questions = question_key in seen_questions
            seen_questions.add(question_key)
            
            is_in_sql_templates = False
            # Check for duplicate SQL template if SQL is available
            # Include hidden_knowledge in the key so that paired interpretations
            # (e.g., ConflictingKnowledge) with the same SQL template but different
            # disambiguation info are not removed as duplicates.
            if question.sql is not None:
                sql_template = mask_sql_values(question.sql)
                hidden = question.hidden_knowledge if isinstance(question, QuestionUnanswerable) else None
                sql_template_key = (sql_template, hidden)
                is_in_sql_templates = sql_template_key in seen_sql_templates
                seen_sql_templates.add(sql_template_key)
            
            # Mark as invalid if duplicate found in either check
            valids.append(not (is_in_questions or is_in_sql_templates))
        return ValidationStageResult(
            stage_name="duplicate_removal",
            validities=valids,
        )
