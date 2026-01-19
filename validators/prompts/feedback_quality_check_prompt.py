from utils.prompt_utils import model_field_descriptions
from dataset_dataclasses.question import Question, QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel, Field
from typing import Annotated, Literal


class FeedbackQualityCheckResponse(BaseModel):
    is_valid: Annotated[Literal["Yes", "No"], Field(description="'Yes' if the feedback correctly and clearly explains why the question is unsolvable, 'No' otherwise. Put only 'Yes' or 'No'.")]


def get_feedback_quality_check_result(response: BaseModel) -> bool:
    """Returns True if feedback is valid, False otherwise."""
    validated = FeedbackQualityCheckResponse.model_validate(response)
    return "yes" in validated.is_valid.strip().lower()


def get_feedback_quality_check_prompt(db: DBDataset, question: Question) -> str:
    """
    Generate a prompt to validate whether the feedback for an unsolvable question
    correctly explains why it cannot be answered.
    """
    assert isinstance(question, QuestionUnanswerable), "Question must be QuestionUnanswerable for feedback validation"
    assert not question.category.is_solvable(), "Question must be unsolvable for feedback validation"
    assert question.hidden_knowledge is not None, "Feedback must be present in hidden_knowledge"
    
    prompt = "You are an expert validator for text-to-SQL question generation. " \
             "Your task is to assess whether the provided feedback correctly explains why a question is fundamentally unsolvable.\n\n"
    
    prompt += "## Context\n"
    prompt += f"**Question:** {question.question}\n"
    
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    
    prompt += "\n## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=3) + "\n\n"
    
    prompt += "## Question Category\n"
    prompt += f"**Category:** {question.category.get_name()}"
    if question.category.get_subname():
        prompt += f" - {question.category.get_subname()}"
    prompt += "\n"
    prompt += f"**Definition:** {question.category.get_definition()}\n\n"
    
    prompt += "## Provided Feedback\n"
    prompt += "This is the feedback that explains why the question is unsolvable:\n"
    prompt += f"**Feedback:** {question.hidden_knowledge}\n\n"
    
    prompt += "## Validation Task\n"
    prompt += "Determine whether the feedback correctly and clearly explains why the question cannot be answered.\n\n"
    
    prompt += "**The feedback should:**\n"
    prompt += "- Correctly identify the type of unsolvability matching the category (missing external knowledge, missing schema elements, improper question, etc.)\n"
    prompt += "- Clearly specify what information, schema elements, or knowledge is missing or problematic\n"
    prompt += "- Be specific enough to be useful (not vague or generic)\n"
    prompt += "- Align with the category definition and the question content\n"
    prompt += "- Explain the fundamental problem that makes the question unanswerable\n\n"
    
    prompt += "**Consider it VALID if:**\n"
    prompt += "- The feedback identifies the correct type of problem for this category\n"
    prompt += "- The explanation is clear, specific, and actionable\n"
    prompt += "- It explains what is missing or wrong without being too vague\n"
    prompt += "- The reasoning aligns with both the question and the category definition\n\n"
    
    prompt += "**Consider it INVALID if:**\n"
    prompt += "- The feedback identifies the wrong type of unsolvability\n"
    prompt += "- The explanation is too vague or generic (e.g., \"missing information\")\n"
    prompt += "- It doesn't specify what exactly is missing or problematic\n"
    prompt += "- The feedback contradicts the category definition\n"
    prompt += "- The feedback suggests the question could be answered with simple clarification (for unsolvable questions)\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a detailed analysis addressing:\n"
    prompt += "1. **Category Alignment:** Does the feedback match the unsolvability category type?\n"
    prompt += "2. **Specificity:** Is the feedback specific enough? What exactly is missing or wrong?\n"
    prompt += "3. **Clarity:** Is the explanation clear and understandable?\n"
    prompt += "4. **Correctness:** Does the feedback correctly identify why this question is fundamentally unsolvable?\n\n"
    
    prompt += "Your reasoning should be concise but thorough (approximately 512 characters).\n\n"
    
    prompt += "Then provide your final verdict as a JSON object with:\n"
    prompt += model_field_descriptions(FeedbackQualityCheckResponse) + "\n\n"
    prompt += "Remember: Focus on whether the feedback correctly and clearly explains the unsolvability."
    
    return prompt
