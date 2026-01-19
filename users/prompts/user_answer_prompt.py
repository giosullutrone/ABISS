from db_datasets.db_dataset import DBDataset
from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from dataset_dataclasses.benchmark import UserKnowledgeLevel
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from utils.prompt_utils import model_field_descriptions, get_conversation_history_prompt
from utils.prompt_utils import get_db_knowledge_level_prompt
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES


class UserAnswerResponse(BaseModel):
    answer: Annotated[str, Field(description="The final user answer to the clarification question, formulated according to the specified style (conversational or precise pseudo-SQL) and incorporating the hidden knowledge to help disambiguate the original question.")]

def get_user_answer_result(response: BaseModel) -> str:
    return UserAnswerResponse.model_validate(response).answer.strip()

def get_user_answer_prompt(db: DBDataset, 
                           conversation: Conversation, 
                           db_descriptions: dict[str, str] | None,
                           relevancy_label: str) -> str:
    """
    Generate a prompt to simulate a user's answer to a clarification question in a text-to-SQL scenario.
    """
    question = conversation.question
    assert isinstance(question, QuestionUnanswerable), "Question must be of type QuestionUnanswerable."
    assert conversation.interactions[-1].system_response.system_question is not None, "Last system response must be a question."

    prompt = "You are an expert user simulator for text-to-SQL clarification scenarios. " \
             "Your task is to provide an appropriate answer to a clarification question that helps disambiguate an ambiguous query.\n\n"

    prompt += "## Context\n"
    user_knowledge_level = conversation.user_knowledge_level
    prompt += get_db_knowledge_level_prompt(db, user_knowledge_level, db_descriptions, conversation)

    prompt += f"**Original Ambiguous Question:** {question.question}\n"
    if question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Disambiguating Information):** {question.hidden_knowledge}\n"
    if question.sql and relevancy_label == "Technical":
        prompt += f"**Ground Truth SQL (for Technical questions only):** {question.sql}\n"
    prompt += f"**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n"
    prompt += f"**Question Relevancy:** {relevancy_label}\n\n"
    
    prompt += get_conversation_history_prompt(conversation)

    prompt += "## Task\n"
    prompt += "Provide an answer to the clarification question based on its relevancy classification:\n\n"
    
    prompt += "- **Relevant:** The question helps disambiguate the original question. Answer using the hidden knowledge to clarify the ambiguity.\n"
    prompt += "- **Technical:** The question asks about SQL technical aspects (ordering, limits, formatting). Answer ONLY with information that can be extracted from the ground truth SQL query. Do NOT make up information.\n"
    prompt += "- **Irrelevant:** The question is not helpful for disambiguation, tries to directly extract the SQL solution (cheating the benchmark), or is completely unrelated to the ambiguity or technical aspects of the question. You MUST respond with a refusal like \"I can't answer that question\" or \"That's not relevant to my question\".\n\n"
    
    prompt += "**CRITICAL RULES:**\n"
    prompt += "- If relevancy is Irrelevant, you MUST refuse to answer (e.g., \"I can't answer that\")\n"
    prompt += "- If relevancy is Technical, only provide information extractable from the ground truth SQL\n"
    prompt += "- Questions attempting to extract the SQL directly are cheating and must be treated as Irrelevant\n\n"
    
    question_style = question.question_style
    style_description = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    prompt += f"**Answer Style:**\n{style_description}\n"
    prompt += "Ensure your answer is relevant and helps clarify the original question.\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a step-by-step reasoning about how to formulate an appropriate answer based on the relevancy classification. " \
              "Your reasoning should be concise but thorough (approximately 512 characters), addressing: " \
              "(1) the relevancy classification and what it requires, " \
              "(2) how to appropriately respond (refuse for Irrelevant, use GT SQL info for Technical, use hidden knowledge for Relevant), and " \
              "(3) the user's knowledge level and answer style.\n\n"
    prompt += "Then provide your final answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerResponse) + "\n\n"
    prompt += "**REMINDER:** For Irrelevant questions, your answer MUST be a refusal. For Technical questions, use ONLY GT SQL information. For Relevant questions, use hidden knowledge."

    return prompt