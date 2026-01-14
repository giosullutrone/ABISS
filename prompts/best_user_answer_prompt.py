from dataset_dataclasses.results import Conversation
from db_datasets.db_dataset import DBDataset
from prompts import get_db_knowledge_level_prompt, get_conversation_history_prompt
from prompts import UserKnowledgeLevel, UserAnswerStyle
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from prompts import model_field_descriptions
from typing import Literal


class BestUserAnswerResponse(BaseModel):
    answer: Annotated[Literal["A", "B"], Field(description="Final selection: 'A' if Answer A is better, 'B' if Answer B is better.")]

def get_best_user_answer_result(response: BaseModel) -> int:
    """Parses the model response and returns '0' or '1' based on whether Answer A is better or B."""
    answer = BestUserAnswerResponse.model_validate(response).answer.strip().upper()
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
    prompt = "You are an expert evaluator for user answers in text-to-SQL clarification scenarios. " \
             "Your task is to select the best user answer that helps disambiguate an ambiguous question.\n\n"
    
    prompt += "## Context\n"
    prompt += get_db_knowledge_level_prompt(db, user_knowledge_level, db_descriptions, conversation)
    prompt += get_conversation_history_prompt(conversation)

    if user_answer_style == UserAnswerStyle.PRECISE:
        prompt += "**Expected Answer Style:** Precise pseudo-SQL manner\n\n"
    else:
        prompt += "**Expected Answer Style:** Natural, conversational manner\n\n"
    
    prompt += "## Candidate Answers\n"
    prompt += f"**Answer A:**\n{generation_a}\n\n"
    prompt += f"**Answer B:**\n{generation_b}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Compare the two candidate answers and determine which one better helps disambiguate the original question. " \
              "Consider the following criteria:\n"
    prompt += "- **Relevance:** Direct addressing of the clarification question\n"
    prompt += "- **Knowledge Integration:** Effective incorporation of hidden knowledge\n"
    prompt += "- **Clarity:** Helpfulness in resolving the ambiguity\n"
    prompt += "- **Style Appropriateness:** Alignment with the user's knowledge level and expected answer style\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a step-by-step analysis comparing the two answers across the evaluation criteria. " \
              "Your reasoning should be concise but thorough (approximately 512 characters), addressing: " \
              "(1) relevance to the clarification question, (2) how well it incorporates the hidden knowledge, " \
              "(3) clarity and helpfulness in resolving the ambiguity, and " \
              "(4) appropriateness to the user's knowledge level and answer style.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(BestUserAnswerResponse) + "\n\n"
    prompt += "Select 'A' or 'B' based on which answer is superior."
    
    return prompt