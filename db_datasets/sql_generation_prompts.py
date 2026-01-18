from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel, Field
from typing import Annotated
from prompts import model_field_descriptions


class SQLGenerationResponse(BaseModel):
    sql: Annotated[str, Field(description="The SQL query that answers the given question based on the database schema.")]


def get_sql_result(response: BaseModel) -> str:
    response = SQLGenerationResponse.model_validate(response)
    return response.sql


# From: https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/gpt_request.py#L99
def generate_comment_prompt(question, knowledge=None):
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above."
    # question_prompt = "-- {}".format(question) + '\n SELECT '
    question_prompt = "-- {}".format(question)
    knowledge_prompt = "-- External Knowledge: {}".format(knowledge)

    if not knowledge_prompt:
        result_prompt = pattern_prompt_no_kg + '\n' + question_prompt
    else:
        result_prompt = knowledge_prompt + '\n' + pattern_prompt_kg + '\n' + question_prompt

    return result_prompt

def get_sql_generation_prompt(db: DBDataset, db_id: str, question: str, evidence: str | None = None) -> str:
    """Generate a prompt for SQL generation with JSON output format."""
    prompt = "You are an expert in converting natural language questions to SQL queries.\n\n"
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(db_id, rows=5) + "\n\n"
    
    prompt += "## Task\n"
    prompt += generate_comment_prompt(question, evidence) + "\n\n"
    
    prompt += "## Instructions\n"
    prompt += "- Use valid SQLite syntax\n"
    prompt += "- Ensure the query correctly uses tables and columns from the schema\n"
    prompt += "- Apply appropriate joins, filters, and aggregations as needed\n"
    prompt += "- The query should be executable and return the correct results\n\n"
    
    prompt += "## Output Format\n"
    prompt += "Provide your SQL query as a JSON object with:\n"
    prompt += model_field_descriptions(SQLGenerationResponse) + "\n\n"
    prompt += "Generate the SQL query now."
    return prompt
