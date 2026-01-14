from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from generators import model_field_descriptions
from typing import Literal
from interactions.prompts import response_debug

first = True

class BestDescriptionResponse(BaseModel):
    thinking_process: Annotated[str, Field(description="Step-by-step reasoning analyzing which description better represents the database schema. "
    "Consider: (1) completeness and coverage of schema elements, "
    "(2) clarity and readability, "
    "(3) accuracy in describing tables, columns, and relationships, and "
    "(4) helpfulness for understanding the database structure. Keep it concise but thorough, about 512 characters.")]
    answer: Annotated[Literal["A", "B"], Field(description="Final selection: 'A' if Description A is better, 'B' if Description B is better.")]

@response_debug
def get_best_description_result(response: str) -> int:
    response_json = BestDescriptionResponse.model_validate_json(response)
    answer = response_json.answer.strip().upper()
    if "A" in answer:
        return 0
    elif "B" in answer:
        return 1
    raise ValueError("Invalid answer in BestDescriptionResponse: must be 'A' or 'B'.")

def get_selection_prompt(db: DBDataset, db_id: str, generation_a: str, generation_b: str) -> str:
    global first
    prompt = f"You are an expert evaluator for database schema descriptions. " \
                "Your task is to select the better natural language description of a database schema.\n\n"
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(db_id, rows=5, db_sql_manipulation=None) + "\n\n"
    
    prompt += "## Candidates\n"
    prompt += f"**Description A:**\n{generation_a}\n\n"
    prompt += f"**Description B:**\n{generation_b}\n\n"
    
    prompt += "## Selection Task\n"
    prompt += "Compare the two candidate descriptions and select which one better represents the database schema. "
    prompt += "Consider:\n"
    prompt += "- Completeness and coverage of schema elements (tables, columns, relationships)\n"
    prompt += "- Clarity and readability of the description\n"
    prompt += "- Accuracy in describing the database structure\n"
    prompt += "- Helpfulness for understanding the schema\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide your evaluation as a JSON object with:\n"
    prompt += model_field_descriptions(BestDescriptionResponse) + "\n\n"
    
    prompt += "Select 'A' or 'B' based on which description is superior."
    if first:
        first = False
        print("Best Description Prompt:", prompt)
    return prompt