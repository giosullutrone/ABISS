from dataset_dataclasses.results import Conversation
from db_datasets.db_dataset import DBDataset
from interactions import get_db_knowledge_level_prompt, get_conversation_history_prompt
from interactions import UserKnowledgeLevel, UserAnswerStyle
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from generators import model_field_descriptions
from typing import Literal
from interactions.prompts import response_debug

first = True

class BestUserAnswerResponse(BaseModel):
    thinking_process: Annotated[str, Field(description="Step-by-step reasoning analyzing which answer better helps disambiguate the original question. "
    "Consider: (1) relevance to the clarification question, "
    "(2) how well it incorporates the hidden knowledge, "
    "(3) clarity and helpfulness in resolving the ambiguity, and "
    "(4) appropriateness to the user's knowledge level and answer style. Keep it concise but thorough, about 512 characters.")]
    answer: Annotated[Literal["A", "B"], Field(description="Final selection: 'A' if Answer A is better, 'B' if Answer B is better.")]

@response_debug
def get_best_user_answer_result(response: str) -> int:
    """Parses the model response and returns '0' or '1' based on whether Answer A is better or B."""
    response_json = BestUserAnswerResponse.model_validate_json(response)
    answer = response_json.answer.strip().upper()
    if "A" in answer:
        return 0
    elif "B" in answer:
        return 1
    raise ValueError("Invalid answer in BestUserAnswerResponse: must contain 'A' or 'B'.")

def get_selection_prompt(db: DBDataset, 
                         db_descriptions: dict[str, str] | None,
                         conversation: Conversation, 
                         generation_a: str, 
                         generation_b: str, 
                         user_knowledge_level: UserKnowledgeLevel, 
                         user_answer_style: UserAnswerStyle) -> str:
    global first
    prompt = f"You are an expert evaluator for user answers in text-to-SQL clarification scenarios. " \
                "Your task is to select the best user answer that helps disambiguate an ambiguous question.\n\n"
    
    prompt += "## Context\n"
    prompt += get_db_knowledge_level_prompt(db, user_knowledge_level, db_descriptions, conversation)
    
    prompt += get_conversation_history_prompt(conversation)

    if user_answer_style == UserAnswerStyle.PRECISE:
        prompt += "The answers to compare should be in a precise pseudo-SQL manner.\n\n"
    else:
        prompt += "The answers to compare should be in a natural, conversational manner.\n\n"
    
    prompt += "## Candidates\n"
    prompt += f"**Answer A:**\n{generation_a}\n\n"
    prompt += f"**Answer B:**\n{generation_b}\n\n"
    
    prompt += "## Selection Task\n"
    prompt += "Compare the two candidate answers and select which one better helps disambiguate the original question. "
    prompt += "Consider:\n"
    prompt += "- Relevance to the clarification question\n"
    prompt += "- Incorporation of hidden knowledge\n"
    prompt += "- Clarity and helpfulness in resolving ambiguity\n"
    prompt += "- Appropriateness to the user's knowledge level and answer style\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide your evaluation as a JSON object with:\n"
    prompt += model_field_descriptions(BestUserAnswerResponse) + "\n\n"
    
    prompt += "Select 'A' or 'B' based on which answer is superior."
    if first:
        first = False
        print("Best User Answer Prompt:", prompt)
    return prompt