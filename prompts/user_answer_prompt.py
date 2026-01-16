from db_datasets.db_dataset import DBDataset
from dataset_dataclasses.results import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from dataset_dataclasses.system import SystemResponseQuestion
from prompts import UserKnowledgeLevel
from prompts import get_db_knowledge_level_prompt, get_conversation_history_prompt
from prompts import STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from prompts import model_field_descriptions


class UserAnswerResponse(BaseModel):
    answer: Annotated[str, Field(description="The final user answer to the clarification question, formulated according to the specified style (conversational or precise pseudo-SQL) and incorporating the hidden knowledge to help disambiguate the original question.")]

def get_user_answer_result(response: BaseModel) -> str:
    return UserAnswerResponse.model_validate(response).answer.strip()

def get_user_answer_prompt(db: DBDataset, 
                           conversation: Conversation, 
                           user_knowledge_level: UserKnowledgeLevel, 
                           db_descriptions: dict[str, str] | None) -> str:
    """
    Generate a prompt to simulate a user's answer to a clarification question in a text-to-SQL scenario.
    """
    question = conversation.question
    assert isinstance(question, QuestionUnanswerable), "Question must be of type QuestionUnanswerable."
    assert isinstance(conversation.interactions[-1].system_response, SystemResponseQuestion), "Last system response must be of type SystemResponseQuestion."

    prompt = "You are an expert user simulator for text-to-SQL clarification scenarios. " \
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
    
    question_style = question.question_style
    style_description = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    prompt += f"**Answer Style:**\n{style_description}\n"
    prompt += "Ensure your answer is relevant and helps clarify the original question.\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a step-by-step reasoning about how to formulate an appropriate answer that helps disambiguate " \
              "the original question using the hidden knowledge. Your reasoning should be concise but thorough " \
              "(approximately 512 characters), considering the user's knowledge level and answer style.\n\n"
    prompt += "Then provide your final answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerResponse) + "\n\n"
    prompt += "Ensure the answer appropriately incorporates the hidden knowledge and follows the specified style."
    
    return prompt