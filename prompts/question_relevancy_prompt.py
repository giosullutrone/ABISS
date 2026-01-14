from dataset_dataclasses.results import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from prompts import model_field_descriptions
from prompts import RelevancyLabel
from typing import Literal


class QuestionRelevancyResponse(BaseModel):
    answer: Annotated[Literal["Relevant", "Technical", "Irrelevant"], Field(description="Final classification: 'Relevant' if the clarification question helps disambiguate the original question using the hidden knowledge, "
    "'Technical' if it focuses on SQL technical aspects unrelated to the ambiguity, or 'Irrelevant' if it doesn't help with disambiguation or tries to extract hidden information.")]

def get_question_relevancy_result(response: BaseModel) -> RelevancyLabel:
    answer = QuestionRelevancyResponse.model_validate(response).answer.strip().lower()
    if 'relevant' in answer:
        return RelevancyLabel.RELEVANT
    elif 'technical' in answer:
        return RelevancyLabel.TECHNICAL
    elif 'irrelevant' in answer:
        return RelevancyLabel.IRRELEVANT
    raise ValueError("Invalid answer in QuestionRelevancyResponse: must contain 'Relevant', 'Technical', or 'Irrelevant'.")

def get_relevancy_prompt(conversation: Conversation) -> str:
    prompt = f"You are an expert evaluator for text-to-SQL clarification question relevance. " \
                "Your task is to assess whether a clarification question helps disambiguate an ambiguous user query.\n\n"
    
    prompt += "## Context\n"
    prompt += f"**Original Ambiguous Question:** {conversation.question.question}\n"
    
    assert isinstance(conversation.question, QuestionUnanswerable), "Question must be of type QuestionUnanswerable."

    if conversation.question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Disambiguating Information):** {conversation.question.hidden_knowledge}\n"
    
    prompt += f"**Clarification Question:** {conversation.interactions[-1]}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Determine if the clarification question is relevant to disambiguating the original question using the hidden knowledge.\n\n"
    
    prompt += "**Relevant:** The clarification question directly addresses the ambiguity in the original question and helps apply the hidden knowledge to resolve it.\n\n"
    
    prompt += "**Technical:** The clarification question is valid but focuses on technical SQL aspects (like ordering, limits, or formatting) that are unrelated to the core ambiguity and hidden knowledge.\n\n"
    
    prompt += "**Irrelevant:** The clarification question does not help resolve the ambiguity, tries to extract the hidden knowledge directly, or is unrelated to the disambiguation task.\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Think step by step before answering, using the following as a guide: Step-by-step reasoning analyzing whether the clarification question is relevant to the original ambiguous question and hidden knowledge. Consider: (1) whether the clarification question addresses the ambiguity in the original question, (2) whether it relates to the hidden knowledge that disambiguates the question, and (3) whether it focuses on disambiguation rather than extracting hidden information or technical SQL details. Keep it concise but thorough, about 512 characters.\n\n"
    prompt += "Provide your evaluation as a JSON object with:\n"
    prompt += model_field_descriptions(QuestionRelevancyResponse) + "\n\n"
    
    prompt += "Choose exactly one classification: 'Relevant', 'Technical', or 'Irrelevant'."
    return prompt
