from utils.prompt_utils import model_field_descriptions
from dataset_dataclasses.question import Question, QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS


class StyleConformanceResponse(BaseModel):
    style_matches: Annotated[Literal["Yes", "No"], Field(description="'Yes' if the question's style matches the specified style requirements, 'No' otherwise. Put only 'Yes' or 'No'.")]


def get_style_conformance_result(response: BaseModel) -> bool:
    """Returns True if the style matches, False otherwise."""
    validated = StyleConformanceResponse.model_validate(response)
    style_valid = "yes" in validated.style_matches.strip().lower()
    return style_valid


def get_style_conformance_prompt(db: DBDataset, question: Question) -> str:
    prompt = "You are an expert in evaluating natural language question styles. " \
             "Your task is to verify whether a generated question's style matches its specified requirements.\n\n"
    
    prompt += "## Question Information\n"
    prompt += f"**Natural Language Question:** {question.question}\n"
    
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += "\n"
    
    # Show the database schema
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=5) + "\n\n"
    
    # Style requirements
    prompt += "## Style Requirement\n"
    prompt += f"**Required Style:** {question.question_style.value}\n"
    prompt += f"{STYLE_DESCRIPTIONS[question.question_style]}\n\n"
    
    prompt += "## Validation Task\n\n"
    
    # Style validation
    prompt += "### Style Validation\n"
    prompt += "Verify that the natural language question matches the specified style:\n"
    prompt += "- Check the vocabulary, tone, and sentence structure\n"
    prompt += "- Compare against the style description provided above\n"
    prompt += "- Ensure the question exhibits the key characteristics of the specified style\n\n"
    
    prompt += "**Answer 'Yes' for style_matches if:** The question clearly follows the specified style requirements.\n"
    prompt += "**Answer 'No' for style_matches if:** The question uses a different style or doesn't match the requirements.\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a detailed analysis addressing:\n"
    prompt += "**Style Analysis:** How well does the question match the specified style? What characteristics support or contradict this?\n\n"

    prompt += "Then provide your final verdict as a JSON object with:\n"
    prompt += model_field_descriptions(StyleConformanceResponse) + "\n\n"
    
    return prompt
