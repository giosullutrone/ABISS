from pydantic import BaseModel, Field
from typing import Annotated
from utils.prompt_utils import model_field_descriptions
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from db_datasets.db_dataset import DBDataset
    from dataset_dataclasses.question import QuestionDifficulty


class SQLGenerationResponse(BaseModel):
    sql: Annotated[str, Field(description="The SQL query that answers the given question based on the database schema.")]


def get_sql_result(response: BaseModel) -> str:
    response = SQLGenerationResponse.model_validate(response)
    return response.sql


def get_sql_generation_prompt(
    db: "DBDataset",
    db_id: str,
    question: str,
    evidence: str | None = None,
    hidden_knowledge: str | None = None,
    question_difficulty: "QuestionDifficulty | None" = None
) -> str:
    """Generate a DIN-SQL-inspired prompt for SQL generation with chain-of-thought reasoning."""
    from dataset_dataclasses.question import QuestionDifficulty

    prompt = "You are an expert SQL developer specializing in text-to-SQL systems.\n\n"

    # Database schema
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(db_id, rows=5) + "\n\n"

    # Question and context
    prompt += "## Question\n"
    prompt += f"{question}\n"
    if evidence:
        prompt += f"\n**Evidence:** {evidence}\n"
    if hidden_knowledge:
        prompt += f"\n**Hidden Knowledge:** {hidden_knowledge}\n"
    prompt += "\n"

    # Difficulty hint
    if question_difficulty is not None:
        prompt += "## Difficulty Hint\n"
        if question_difficulty == QuestionDifficulty.SIMPLE:
            prompt += "Write a simple query. No JOINs or nested queries should be needed.\n\n"
        elif question_difficulty == QuestionDifficulty.MODERATE:
            prompt += "The query may require JOINs across multiple tables.\n\n"
        else:
            prompt += ("The query requires nested subqueries, set operations "
                       "(UNION/INTERSECT/EXCEPT), or complex aggregations. "
                       "Decompose the question into sub-questions first.\n\n")

    # Chain-of-thought instructions
    prompt += "## Chain-of-Thought Instructions\n"
    prompt += "Follow these steps to generate the SQL query:\n\n"

    is_hard = question_difficulty in (QuestionDifficulty.COMPLEX, QuestionDifficulty.HIGHLY_COMPLEX) if question_difficulty else False

    prompt += "**Step 1 - Schema Linking:**\n"
    prompt += ("Identify the relevant tables, columns, and foreign key relationships "
               "for this question. List them explicitly.\n\n")

    if is_hard:
        prompt += "**Step 2 - Sub-question Decomposition:**\n"
        prompt += ("Break the question into simpler sub-questions. For each sub-question, "
                   "identify the SQL operation needed.\n\n")

    prompt += f"**Step {'3' if is_hard else '2'} - SQL Generation:**\n"
    prompt += "Write the final SQL query following these rules:\n"
    prompt += "- Use valid SQLite syntax\n"
    prompt += "- Use double quotes for identifiers with spaces or special characters\n"
    prompt += "- Use explicit aliases for aggregated or calculated columns (e.g., `SELECT COUNT(*) AS count ...`)\n"
    prompt += "- The query must be executable and return the correct results\n"
    if hidden_knowledge:
        prompt += "- The SQL must faithfully represent the interpretation described in the hidden knowledge\n"
    prompt += "\n"

    # Output format
    prompt += "## Output Format\n"
    prompt += "Provide your SQL query as a JSON object with:\n"
    prompt += model_field_descriptions(SQLGenerationResponse) + "\n\n"
    prompt += "Generate the SQL query now."
    return prompt
