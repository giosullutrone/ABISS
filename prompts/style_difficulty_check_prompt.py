from prompts import model_field_descriptions
from dataset_dataclasses.question import Question, QuestionUnanswerable, QuestionStyle, QuestionDifficulty
from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from prompts.generator_style_and_difficulty_prompt import STYLE_DESCRIPTIONS, DIFFICULTY_CRITERIA


class StyleDifficultyCheckResponse(BaseModel):
    style_matches: Annotated[Literal["Yes", "No"], Field(description="'Yes' if the question's style matches the specified style requirements, 'No' otherwise. Put only 'Yes' or 'No'.")]
    difficulty_matches: Annotated[Literal["Yes", "No"], Field(description="'Yes' if the SQL query's complexity matches the specified difficulty level, 'No' otherwise. Put only 'Yes' or 'No'.")]


def get_style_difficulty_validation_result(response: BaseModel) -> bool:
    """Returns True if both style and difficulty match, False otherwise."""
    validated = StyleDifficultyCheckResponse.model_validate(response)
    style_valid = "yes" in validated.style_matches.strip().lower()
    difficulty_valid = "yes" in validated.difficulty_matches.strip().lower()
    return style_valid and difficulty_valid


def get_style_difficulty_validation_prompt(db: DBDataset, question: Question) -> str:
    prompt = "You are an expert in evaluating natural language question styles and SQL query complexity. " \
             "Your task is to verify whether a generated question's style and SQL difficulty match their specified requirements.\n\n"
    
    prompt += "## Question Information\n"
    prompt += f"**Natural Language Question:** {question.question}\n"
    
    # Show the database schema
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=5) + "\n\n"
    
    # Style requirements
    prompt += "## Style Requirement\n"
    prompt += f"**Required Style:** {question.question_style.value}\n"
    prompt += f"{STYLE_DESCRIPTIONS[question.question_style]}\n\n"
    
    # Difficulty requirements and SQL
    prompt += "## Difficulty Requirement\n"
    prompt += f"**Required Difficulty:** {question.question_difficulty.value}\n"
    prompt += f"{DIFFICULTY_CRITERIA[question.question_difficulty]}\n\n"
    
    if question.sql:
        prompt += "## Generated SQL Query\n"
        if isinstance(question, QuestionUnanswerable) and question.is_solvable:
            prompt += f"**SQL Query:** {question.sql}\n"
            if question.hidden_knowledge:
                prompt += f"**Context:** This SQL represents one interpretation knowing that: {question.hidden_knowledge}\n"
        else:
            prompt += f"**SQL Query:** {question.sql}\n"
        
        # Try to execute and show results
        results = db.execute_query(db_id=question.db_id, sql_query=question.sql)
        if results is not None:
            prompt += f"**Query Results (first 5 rows):** {results[:5]}\n"
        prompt += "\n"
    else:
        prompt += "## SQL Query\n"
        prompt += "**Note:** This question is unanswerable with the current schema (no SQL provided).\n"
        prompt += "For difficulty validation, consider what SQL complexity would be required if the schema were complete.\n\n"
    
    prompt += "## Validation Tasks\n\n"
    
    # Style validation
    prompt += "### Task 1: Style Validation\n"
    prompt += "Verify that the natural language question matches the specified style:\n"
    prompt += "- Check the vocabulary, tone, and sentence structure\n"
    prompt += "- Compare against the style description and example provided above\n"
    prompt += "- Ensure the question exhibits the key characteristics of the specified style\n\n"
    
    prompt += "**Answer 'Yes' for style_matches if:** The question clearly follows the specified style requirements.\n"
    prompt += "**Answer 'No' for style_matches if:** The question uses a different style or doesn't match the requirements.\n\n"
    
    # Difficulty validation
    prompt += "### Task 2: Difficulty Validation\n"
    if question.sql:
        prompt += "Verify that the SQL query's complexity matches the specified difficulty level:\n"
        prompt += "- Analyze the SQL structure (joins, subqueries, CTEs, window functions, etc.)\n"
        prompt += "- Compare against the difficulty criteria provided above\n"
        prompt += "- Ensure the query complexity aligns with the specified difficulty characteristics\n\n"
        
        prompt += "**Answer 'Yes' for difficulty_matches if:** The SQL query's complexity clearly matches the specified difficulty level.\n"
        prompt += "**Answer 'No' for difficulty_matches if:** The SQL is too simple or too complex for the specified difficulty.\n\n"
    else:
        prompt += "Since no SQL is provided (unanswerable question), verify that the question's scope and complexity align with the specified difficulty:\n"
        prompt += "- Consider what SQL would be needed if the schema were complete\n"
        prompt += "- Assess whether the question's complexity matches the difficulty level\n"
        prompt += "- Check if the question would require the types of SQL features described in the difficulty criteria\n\n"
        
        prompt += "**Answer 'Yes' for difficulty_matches if:** The question's implied SQL complexity matches the specified difficulty.\n"
        prompt += "**Answer 'No' for difficulty_matches if:** The question's scope is too simple or too complex for the specified difficulty.\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a detailed analysis addressing:\n"
    prompt += "1. **Style Analysis:** How well does the question match the specified style? What characteristics support or contradict this?\n"
    prompt += "2. **Difficulty Analysis:** "
    if question.sql:
        prompt += "Does the SQL query's complexity match the specified difficulty? What SQL features support this assessment?\n\n"
    else:
        prompt += "Does the question's scope and implied SQL complexity match the specified difficulty?\n\n"
    
    prompt += "Then provide your final verdict as a JSON object with:\n"
    prompt += model_field_descriptions(StyleDifficultyCheckResponse) + "\n\n"
    
    return prompt
