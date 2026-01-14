from db_datasets.db_dataset import DBDataset
from prompts import model_field_descriptions
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from typing import Literal


class CheckGTResponse(BaseModel):
    answer: Annotated[Literal["Yes", "No"], Field(description="Final verdict: 'Yes' if the SQL query correctly answers the disambiguated question (considering the hidden knowledge), "
    "or 'No' if it fails to capture the intended meaning or contains errors.")]

def get_gt_validation_result(response: BaseModel) -> bool:
    answer = CheckGTResponse.model_validate(response).answer.strip().lower()
    return "yes" in answer

def get_gt_validation_prompt(db: DBDataset, question: QuestionUnanswerable) -> str:
    prompt = "You are an expert SQL validator for text-to-SQL benchmarks. " \
             "Your task is to verify whether a ground truth SQL query correctly answers a natural language question " \
             "after disambiguation through hidden knowledge.\n\n"
    
    prompt += "## Context\n"
    prompt += "The original question was ambiguous or underspecified. The user has provided hidden knowledge " \
                "that clarifies their intent. Your job is to evaluate whether the provided SQL query correctly " \
                "implements this disambiguated intent.\n\n"
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=5) + "\n\n"
    
    prompt += "## Question Information\n"
    prompt += f"**Natural Language Question:** {question.question}\n"
    
    if question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Disambiguating Information):** {question.hidden_knowledge}\n"
    
    prompt += f"**Ground Truth SQL Query:** {question.sql}\n"
    
    assert question.sql is not None, "GT SQL query is None."
    results = db.execute_query(
        db_id=question.db_id, 
        sql_query=question.sql
    )
    if results is not None:
        prompt += f"**Query Execution Results (first 5 rows):** {results[:5]}\n\n"
    else:
        prompt += "**Query Execution Results:** Query execution returned no results.\n\n"
    
    prompt += "## Validation Task\n"
    prompt += "Evaluate whether the SQL query correctly answers the question given the hidden knowledge. "
    prompt += "Answer **Yes** if:\n"
    prompt += "- The SQL correctly uses tables and columns from the schema\n"
    prompt += "- The query logic (joins, WHERE clauses, aggregations, etc.) matches the disambiguated intent\n"
    prompt += "- The results align with what the question asks for\n"
    prompt += "- The hidden knowledge is properly reflected in the SQL structure\n\n"
    
    prompt += "Answer **No** if:\n"
    prompt += "- The SQL uses incorrect or non-existent schema elements\n"
    prompt += "- The query logic doesn't match the question's requirements\n"
    prompt += "- The hidden knowledge is ignored or incorrectly applied\n"
    prompt += "- The query would return incorrect or incomplete results\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Think step by step before answering, using the following as a guide: Step-by-step reasoning analyzing whether the SQL query correctly implements the disambiguated intent. Consider: (1) whether the query uses the correct tables and columns from the schema, (2) whether the joins, filters, and aggregations match the question's requirements given the hidden knowledge, and (3) whether the query results align with what the question is asking for.\n\n"
    prompt += "Provide your evaluation as a JSON object with:\n"
    prompt += model_field_descriptions(CheckGTResponse) + "\n\n"
    return prompt
