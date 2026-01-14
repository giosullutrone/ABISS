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
    prompt = "You are an expert at describing database schemas in natural language. " \
             "Your task is to analyze a database schema and provide a comprehensive description.\n\n"
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(db_id, rows=5) + "\n\n"
    
    prompt += "## Task\n"
    prompt += "Analyze the database schema and generate a natural language description that covers:\n"
    prompt += "- **Main Tables:** Purpose and role of each table\n"
    prompt += "- **Key Columns:** Important columns and their data types\n"
    prompt += "- **Relationships:** Foreign keys and how tables connect\n"
    prompt += "- **Constraints & Features:** Any constraints, indexes, or notable characteristics\n"
    prompt += "- **Overall Structure:** The database's organization and intended purpose\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a step-by-step analysis of the database schema structure. " \
              "Your reasoning should be concise but thorough (approximately 512 characters), covering: " \
              "tables, columns, relationships, and key features that should be included in the description.\n\n"
    prompt += "Then provide your final description as a JSON object with:\n"
    prompt += model_field_descriptions(SchemaToNLResponse) + "\n\n"
    prompt += "Ensure the description is clear, comprehensive, and suitable for understanding the database structure."
    
    return prompt
