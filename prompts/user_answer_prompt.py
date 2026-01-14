from db_datasets.db_dataset import DBDataset
from dataset_dataclasses.results import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from dataset_dataclasses.system import SystemResponseQuestion
from prompts import UserAnswerStyle, UserKnowledgeLevel
from prompts import get_db_knowledge_level_prompt, get_conversation_history_prompt
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from prompts import model_field_descriptions


class UserAnswerResponse(BaseModel):
    thinking_process: Annotated[str, Field(description="Step-by-step reasoning about how to formulate an appropriate answer that helps disambiguate the original question using the hidden knowledge, considering the user's knowledge level and answer style. Keep it concise but thorough, about 512 characters.")]
    answer: Annotated[str, Field(description="The final user answer to the clarification question, formulated according to the specified style (conversational or precise pseudo-SQL) and incorporating the hidden knowledge to help disambiguate the original question.")]

def get_user_answer_result(response: str) -> str:
    response_json = UserAnswerResponse.model_validate_json(response)
    return response_json.answer.strip()

def get_user_answer_prompt(db: DBDataset, 
                           conversation: Conversation, 
                           user_knowledge_level: UserKnowledgeLevel, 
                           user_answer_style: UserAnswerStyle,
                           db_descriptions: dict[str, str] | None) -> str:
    """
    Precise prompt to get the user answer to the clarification question with a conversational style.
    """
    question = conversation.question
    assert isinstance(question, QuestionUnanswerable), "Question must be of type QuestionUnanswerable."
    assert isinstance(conversation.interactions[-1].system_response, SystemResponseQuestion), "Last system response must be of type SystemResponseQuestion."

    prompt = f"You are an expert user simulator for text-to-SQL clarification scenarios. " \
                "Your task is to provide an appropriate answer to a clarification question that helps disambiguate an ambiguous query.\n\n"

    prompt += "## Context\n"
    prompt += get_db_knowledge_level_prompt(db, user_knowledge_level, db_descriptions, conversation)

    prompt += f"**Original Ambiguous Question:** {question.question}\n"
    if question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Disambiguating Information):** {question.hidden_knowledge}\n"
    prompt += f"**Clarification Question:** {conversation.interactions[-1].system_response.question}\n\n"
    
    prompt += get_conversation_history_prompt(conversation)

    prompt += "## Task\n"
    prompt += "Provide an answer to the clarification question that helps disambiguate the original question using the hidden knowledge.\n\n"
    
    if user_answer_style == UserAnswerStyle.CONVERSATIONAL:
        prompt += "**Style:** Respond in a natural, conversational manner as if you were speaking directly to the text-to-SQL system. Make sure your answer is relevant and helps clarify the original question.\n\n"
    else:
        prompt += "**Style:** Respond in a precise pseudo-SQL manner, focusing on providing the necessary information to clarify the original question. Make sure your answer is relevant and helps clarify the original question.\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide your response as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerResponse) + "\n\n"
    
    prompt += "Ensure the answer appropriately incorporates the hidden knowledge and follows the specified style."
    return prompt