from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from utils.prompt_utils import model_field_descriptions


class FeedbackEvaluationResponse(BaseModel):
    matches: Annotated[Literal["Yes", "No"], Field(description="'Yes' if the system's feedback correctly identifies why the question is unsolvable and matches the expected feedback, 'No' otherwise. Put only 'Yes' or 'No'.")]


def get_feedback_evaluation_result(response: BaseModel) -> bool:
    """Returns True if feedback matches, False otherwise."""
    validated = FeedbackEvaluationResponse.model_validate(response)
    return "yes" in validated.matches.strip().lower()


def get_feedback_evaluation_prompt(conversation: Conversation) -> str:
    """
    Generate a prompt to evaluate whether the system's feedback correctly identifies
    why an unsolvable question cannot be answered.
    """
    question = conversation.question
    assert isinstance(question, QuestionUnanswerable), "Question must be QuestionUnanswerable for feedback evaluation"
    assert not question.category.is_solvable(), "Question must be unsolvable for feedback evaluation"
    assert question.hidden_knowledge is not None, "Expected feedback must be present in hidden_knowledge"
    
    system_feedback = conversation.predicted_feedback
    assert system_feedback is not None, "System must have provided feedback"
    
    prompt = "You are an expert evaluator for text-to-SQL systems. " \
             "Your task is to assess whether the system correctly identified why a question is fundamentally unsolvable.\n\n"
    
    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    
    prompt += "## Question Category\n"
    prompt += f"**Category:** {question.category.get_name()}"
    if question.category.get_subname():
        prompt += f" - {question.category.get_subname()}"
    prompt += "\n"
    prompt += f"**Definition:** {question.category.get_definition()}\n\n"
    
    prompt += "## Expected Feedback\n"
    prompt += "This is the expected explanation of why the question is unsolvable:\n"
    prompt += f"**Expected:** {question.hidden_knowledge}\n\n"
    
    prompt += "## System's Feedback\n"
    prompt += "This is what the system provided as feedback:\n"
    prompt += f"**System's Response:** {system_feedback}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Determine whether the system's feedback correctly identifies why the question is unsolvable.\n\n"
    
    prompt += "**The feedback should:**\n"
    prompt += "- Correctly identify the type of unsolvability (missing external knowledge, missing schema elements, improper question, etc.)\n"
    prompt += "- Explain what specific information, schema elements, or knowledge is missing or problematic\n"
    prompt += "- Match the core reasoning present in the expected feedback\n"
    prompt += "- Be semantically equivalent even if worded differently\n\n"
    
    prompt += "**Consider it a match if:**\n"
    prompt += "- The system identifies the same fundamental problem as the expected feedback\n"
    prompt += "- The key missing elements or issues are the same (exact wording doesn't need to match)\n"
    prompt += "- The reasoning aligns with the question's category and definition\n\n"
    
    prompt += "**Consider it NOT a match if:**\n"
    prompt += "- The system identifies a different reason for unsolvability\n"
    prompt += "- The system misses the key problem entirely\n"
    prompt += "- The feedback is too vague or generic to be useful\n"
    prompt += "- The system's explanation contradicts the expected feedback\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a detailed analysis addressing:\n"
    prompt += "1. **Expected Feedback Analysis:** What specific problem does the expected feedback identify?\n"
    prompt += "2. **System Feedback Analysis:** What does the system's feedback claim is the problem?\n"
    prompt += "3. **Comparison:** Do they identify the same fundamental issue? Are the key elements the same?\n"
    prompt += "4. **Alignment with Category:** Does the system's feedback align with the question's unsolvability category?\n\n"
    
    prompt += "Your reasoning should be concise but thorough (approximately 512 characters).\n\n"
    
    prompt += "Then provide your final verdict as a JSON object with:\n"
    prompt += model_field_descriptions(FeedbackEvaluationResponse) + "\n\n"
    prompt += "Remember: Focus on semantic equivalence, not exact wording. The system should identify the same fundamental problem."
    
    return prompt
