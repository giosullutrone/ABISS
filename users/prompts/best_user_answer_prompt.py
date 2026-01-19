from dataset_dataclasses.benchmark import Conversation
from db_datasets.db_dataset import DBDataset
from utils.prompt_utils import get_conversation_history_prompt
from utils.prompt_utils import get_db_knowledge_level_prompt
from dataset_dataclasses.benchmark import UserKnowledgeLevel
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from utils.prompt_utils import model_field_descriptions
from typing import Literal


class BestUserAnswerResponse(BaseModel):
    answer: Annotated[Literal["A", "B"], Field(description="Final selection: 'A' if Answer A is better, 'B' if Answer B is better. Put only 'A' or 'B'.")]


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
                         relevancy_label: str) -> str:
    prompt = "You are an expert evaluator for user answers in text-to-SQL clarification scenarios. " \
             "Your task is to select the best user answer that helps disambiguate an ambiguous question.\n\n"
    
    prompt += "## Context\n"
    user_knowledge_level = conversation.user_knowledge_level
    prompt += get_db_knowledge_level_prompt(db, user_knowledge_level, db_descriptions, conversation)
    prompt += get_conversation_history_prompt(conversation)
    
    # Add ground truth SQL for Technical questions
    if relevancy_label == "Technical" and conversation.question.sql:
        prompt += f"**Ground Truth SQL (for Technical questions):** {conversation.question.sql}\n"
    
    prompt += f"**Question Relevancy:** {relevancy_label}\n\n"

    question_style = conversation.question.question_style
    style_description = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    prompt += f"**Expected Answer Style:**\n{style_description}\n\n"
    
    prompt += "## Candidate Answers\n"
    prompt += f"**Answer A:**\n{generation_a}\n\n"
    prompt += f"**Answer B:**\n{generation_b}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Compare the two candidate answers based on the question's relevancy classification:\n\n"
    
    prompt += "**For Relevant Questions:**\n"
    prompt += "- Both answers should use hidden knowledge to disambiguate\n"
    prompt += "- Select the answer that better clarifies the ambiguity\n\n"
    
    prompt += "**For Technical Questions:**\n"
    prompt += "- Both answers should only use information from the ground truth SQL\n"
    prompt += "- Select the answer that better addresses the technical aspect\n\n"
    
    prompt += "**For Irrelevant Questions:**\n"
    prompt += "- CRITICAL: Both answers MUST refuse to answer (e.g., \"I can't answer that\")\n"
    prompt += "- Irrelevant includes: SQL extraction attempts (cheating), unhelpful questions, or completely unrelated questions\n"
    prompt += "- If one answer refuses and the other doesn't, the refusing answer is AUTOMATICALLY better\n"
    prompt += "- If both refuse, or both fail to refuse, choose Answer A (first answer)\n\n"
    
    prompt += "**General Evaluation Criteria:**\n"
    prompt += "- **Relevance:** Appropriate response based on relevancy classification\n"
    prompt += "- **Correctness:** For Irrelevant questions, refusing to answer is the only correct response\n"
    prompt += "- **Knowledge Integration:** For Relevant questions, effective use of hidden knowledge\n"
    prompt += "- **Clarity:** Helpfulness in resolving the appropriate type of ambiguity\n"
    prompt += "- **Style Appropriateness:** Alignment with the user's knowledge level and expected answer style\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a step-by-step analysis comparing the two answers. " \
              "Your reasoning should be concise but thorough (approximately 512 characters), addressing: " \
              "(1) whether answers are appropriate for the relevancy classification (especially for Irrelevant questions where refusal is mandatory), " \
              "(2) how well they address the clarification question, " \
              "(3) correctness and clarity, and " \
              "(4) appropriateness to the user's knowledge level and answer style.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(BestUserAnswerResponse) + "\n\n"
    prompt += "**CRITICAL:** For Irrelevant questions, an answer that refuses is always better than one that doesn't. " \
              "In case of a tie (both refuse or both don't), select Answer A."
    
    return prompt