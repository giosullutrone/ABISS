from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel, Field
from typing import Annotated
from utils.prompt_utils import model_field_descriptions
from dataset_dataclasses.question import Question
from categories.category import Category


class CategoryClassificationResponse(BaseModel):
    category_name: Annotated[str, Field(description="The name of the category that best describes this question. Must be one of the valid category names.")]
    category_subname: Annotated[str | None, Field(description="The subname of the category if applicable, otherwise null.")]


def get_category_result(response: BaseModel) -> tuple[str, str | None]:
    validated = CategoryClassificationResponse.model_validate(response)
    return (validated.category_name, validated.category_subname)


def get_category_classification_prompt(db: DBDataset, question: Question, categories: list[Category]) -> str:
    """Generate a prompt for classifying a question into a category."""
    
    prompt = "You are an expert in analyzing natural language questions for text-to-SQL systems. " \
             "Your task is to classify a question into one of several categories based on whether it can be " \
             "directly answered with a SQL query or requires clarification/additional information.\n\n"
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=5) + "\n\n"
    
    prompt += "## Question to Classify\n"
    prompt += f"**Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += "\n"
    
    prompt += "## Available Categories\n"
    prompt += "Analyze the question and classify it into ONE of the following categories:\n\n"
    for category in categories:
        prompt += f"**{category.get_name()}**"
        if category.get_subname():
            prompt += f" ({category.get_subname()})"
        prompt += f"\n{category.get_definition()}\n\n"
    
    prompt += "## Classification Task\n"
    prompt += "Carefully analyze the question and determine which category it belongs to. Consider:\n"
    prompt += "- Can the question be directly answered with the given schema?\n"
    prompt += "- Is there any ambiguity or missing information?\n"
    prompt += "- If ambiguous, can it be resolved through user interaction?\n"
    prompt += "- If information is missing, is it schema-related or external knowledge?\n\n"
    
    prompt += "## Output Format\n"
    prompt += "Provide your classification as a JSON object with:\n"
    prompt += model_field_descriptions(CategoryClassificationResponse) + "\n\n"
    prompt += "Classify the question now."
    
    return prompt
