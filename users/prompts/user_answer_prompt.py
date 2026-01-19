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
    assert conversation.interactions[-1].system_response.system_question is not None, "Last system response must be a question."

    prompt = "You are an expert user simulator for text-to-SQL clarification scenarios. " \
             "Your task is to provide an appropriate answer to a clarification question that helps disambiguate an ambiguous query.\n\n"

    prompt += "## Context\n"
    user_knowledge_level = conversation.user_knowledge_level
    prompt += get_db_knowledge_level_prompt(db, user_knowledge_level, db_descriptions, conversation)

    prompt += f"**Original Ambiguous Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    
    # Determine question type
    is_answerable = question.category.is_answerable()
    is_solvable = question.category.is_solvable() if isinstance(question, QuestionUnanswerable) else True
    
    # Show hidden knowledge only for solvable questions (answerable or unanswerable-solvable)
    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge and is_solvable:
        prompt += f"**Hidden Knowledge (Disambiguating Information):** {question.hidden_knowledge}\n"
    
    if question.sql and relevancy_label == "Technical":
        prompt += f"**Ground Truth SQL (for Technical questions only):** {question.sql}\n"
    
    # Indicate question type to guide user behavior
    if is_answerable:
        prompt += f"**Question Type:** Answerable (no disambiguation needed)\n"
    elif is_solvable:
        prompt += f"**Question Type:** Unanswerable but Solvable (requires disambiguation through interaction)\n"
    else:
        prompt += f"**Question Type:** Unsolvable (cannot be answered even with clarification - missing external knowledge, improper question, or missing schema elements)\n"
    
    prompt += f"**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n"
    prompt += f"**Question Relevancy:** {relevancy_label}\n\n"
    
    prompt += get_conversation_history_prompt(conversation)

    prompt += "## Task\n"
    prompt += "Provide an answer to the clarification question based on its relevancy classification and question type:\n\n"
    
    prompt += "**For Answerable or Unanswerable-Solvable questions:**\n"
    prompt += "- **Relevant:** The question helps disambiguate. Answer using the hidden knowledge to clarify the ambiguity.\n"
    prompt += "- **Technical:** Answer ONLY with information from the ground truth SQL. Do NOT make up information.\n"
    prompt += "- **Irrelevant:** MUST refuse (e.g., \"I can't answer that\").\n\n"
    
    prompt += "**For Unsolvable questions:**\n"
    prompt += "- The question is fundamentally unsolvable (missing external knowledge/schema elements, or improper question).\n"
    prompt += "- You should answer naturally without revealing the specific reason why it's unsolvable.\n"
    prompt += "- For Relevant questions about the ambiguity: provide natural clarifications but the question remains unsolvable.\n"
    prompt += "- For Irrelevant questions: MUST refuse to answer.\n"
    prompt += "- Do NOT explicitly state \"this question is unsolvable\" - answer as a normal user would.\n\n"
    
    prompt += "**CRITICAL RULES:**\n"
    prompt += "- If relevancy is Irrelevant, you MUST refuse to answer (e.g., \"I can't answer that\")\n"
    prompt += "- If relevancy is Technical, only provide information extractable from the ground truth SQL\n"
    prompt += "- For Unsolvable questions, do NOT reveal why it's unsolvable - the system should discover this\n"
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