from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from prompts import model_field_descriptions


class SchemaToNLResponse(BaseModel):
    description: Annotated[str, Field(description="A concise but comprehensive natural language summary of the database schema, covering all important tables, their relationships, and notable features.")]

def get_schema_to_nl_result(response: BaseModel) -> str:
    return SchemaToNLResponse.model_validate(response).description.strip()

def get_generation_prompt(db: DBDataset, db_id: str) -> str:
    prompt = f"You are an expert at describing database schemas in natural language. " \
                "Your task is to analyze a database schema and provide a comprehensive description.\n\n"
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(db_id, rows=5, db_sql_manipulation=None) + "\n\n"
    
    prompt += "## Task\n"
    prompt += "Analyze the database schema and generate a natural language description that covers:\n"
    prompt += "- Main tables and their purposes\n"
    prompt += "- Key columns and their data types\n"
    prompt += "- Relationships between tables (foreign keys, joins)\n"
    prompt += "- Any constraints, indexes, or notable features\n"
    prompt += "- Overall database structure and purpose\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Think step by step before answering, using the following as a guide: Step-by-step reasoning analyzing the database schema structure, including tables, columns, relationships, and key features that should be described. Keep it concise but thorough, about 512 characters.\n\n"
    prompt += "Provide your analysis as a JSON object with:\n"
    prompt += model_field_descriptions(SchemaToNLResponse) + "\n\n"
    
    prompt += "Ensure the description is clear, comprehensive, and suitable for understanding the database structure."
    return prompt
