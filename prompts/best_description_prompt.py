from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from prompts import model_field_descriptions
from typing import Literal


class BestDescriptionResponse(BaseModel):
    answer: Annotated[Literal["A", "B"], Field(description="Final selection: 'A' if Description A is better, 'B' if Description B is better.")]

def get_best_description_result(response: BaseModel) -> int:
    answer = BestDescriptionResponse.model_validate(response).answer.strip().upper()
    if "A" in answer:
        return 0
    elif "B" in answer:
        return 1
    raise ValueError("Invalid answer in BestDescriptionResponse: must be 'A' or 'B'.")

def get_selection_prompt(db: DBDataset, db_id: str, generation_a: str, generation_b: str) -> str:
    prompt = "You are an expert evaluator for database schema descriptions. " \
             "Your task is to select the better natural language description of a database schema.\n\n"
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(db_id, rows=5) + "\n\n"
    
    prompt += "## Candidate Descriptions\n"
    prompt += f"**Description A:**\n{generation_a}\n\n"
    prompt += f"**Description B:**\n{generation_b}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Compare the two candidate descriptions and determine which one better represents the database schema. " \
              "Consider the following criteria:\n"
    prompt += "- **Completeness:** Coverage of all important schema elements (tables, columns, relationships)\n"
    prompt += "- **Clarity:** Readability and ease of understanding\n"
    prompt += "- **Accuracy:** Correctness in describing the database structure\n"
    prompt += "- **Usefulness:** Helpfulness for understanding the schema's purpose and organization\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a step-by-step analysis comparing the two descriptions across the evaluation criteria. " \
              "Your reasoning should be concise but thorough (approximately 512 characters), addressing: " \
              "(1) completeness and coverage of schema elements, (2) clarity and readability, " \
              "(3) accuracy in describing tables, columns, and relationships, and " \
              "(4) usefulness for understanding the database structure.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(BestDescriptionResponse) + "\n\n"
    prompt += "Select 'A' or 'B' based on which description is superior."
    
    return prompt