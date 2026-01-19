from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
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
    
    # Determine question type
    question = conversation.question
    is_answerable = question.category.is_answerable()
    is_solvable = question.category.is_solvable() if isinstance(question, QuestionUnanswerable) else True
    
    # Show hidden knowledge only for solvable questions
    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge and is_solvable:
        prompt += f"**Hidden Knowledge (Disambiguating Information):** {question.hidden_knowledge}\n"
    
    # Add ground truth SQL for Technical questions
    if relevancy_label == "Technical" and question.sql:
        prompt += f"**Ground Truth SQL (for Technical questions):** {question.sql}\n"
    
    # Indicate question type
    if is_answerable:
        prompt += f"**Question Type:** Answerable (no disambiguation needed)\n"
    elif is_solvable:
        prompt += f"**Question Type:** Unanswerable but Solvable (requires disambiguation)\n"
    else:
        prompt += f"**Question Type:** Unsolvable (cannot be answered - missing external knowledge, improper question, or missing schema elements)\n"
    
    prompt += f"**Question Relevancy:** {relevancy_label}\n\n"

    question_style = question.question_style
    style_description = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    prompt += f"**Expected Answer Style:**\n{style_description}\n\n"
    
    prompt += "## Candidate Answers\n"
    prompt += f"**Answer A:**\n{generation_a}\n\n"
    prompt += f"**Answer B:**\n{generation_b}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Compare the two candidate answers based on the question type and relevancy classification:\n\n"
    
    prompt += "**For Answerable or Unanswerable-Solvable Questions:**\n\n"
    
    prompt += "- **Relevant Questions:**\n"
    prompt += "  - Both answers should use hidden knowledge to disambiguate\n"
    prompt += "  - Select the answer that better clarifies the ambiguity\n\n"
    
    prompt += "- **Technical Questions:**\n"
    prompt += "  - Both answers should only use information from the ground truth SQL\n"
    prompt += "  - Select the answer that better addresses the technical aspect\n\n"
    
    prompt += "- **Irrelevant Questions:**\n"
    prompt += "  - CRITICAL: Both answers MUST refuse to answer (e.g., \"I can't answer that\")\n"
    prompt += "  - If one answer refuses and the other doesn't, the refusing answer is AUTOMATICALLY better\n"
    prompt += "  - If both refuse, or both fail to refuse, choose Answer A\n\n"
    
    prompt += "**For Unsolvable Questions:**\n\n"
    
    prompt += "- **Relevant Questions:**\n"
    prompt += "  - Answers should provide natural clarifications without revealing why the question is unsolvable\n"
    prompt += "  - Select the answer that sounds more natural and helpful while maintaining the unsolvability\n"
    prompt += "  - Do NOT prefer answers that explicitly state \"this is unsolvable\"\n\n"
    
    prompt += "- **Technical Questions:**\n"
    prompt += "  - Both answers should only use information from the ground truth SQL (if available)\n"
    prompt += "  - Select the answer that better addresses the technical aspect\n\n"
    
    prompt += "- **Irrelevant Questions:**\n"
    prompt += "  - CRITICAL: Both answers MUST refuse to answer\n"
    prompt += "  - If one answer refuses and the other doesn't, the refusing answer is AUTOMATICALLY better\n"
    prompt += "  - If both refuse, or both fail to refuse, choose Answer A\n\n"
    
    prompt += "**General Evaluation Criteria:**\n"
    prompt += "- **Relevance:** Appropriate response based on question type and relevancy classification\n"
    prompt += "- **Correctness:** For Irrelevant questions, refusing is mandatory; for Unsolvable questions, don't reveal unsolvability\n"
    prompt += "- **Knowledge Integration:** For solvable questions, effective use of hidden knowledge\n"
    prompt += "- **Naturalness:** For Unsolvable questions, answers should sound natural without revealing the problem\n"
    prompt += "- **Clarity:** Helpfulness in providing appropriate responses\n"
    prompt += "- **Style Appropriateness:** Alignment with the user's knowledge level and expected answer style\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a step-by-step analysis comparing the two answers. " \
              "Your reasoning should be concise but thorough (approximately 512 characters), addressing: " \
              "(1) question type (answerable/solvable/unsolvable) and how it affects evaluation, " \
              "(2) whether answers are appropriate for the relevancy classification, " \
              "(3) for Irrelevant: whether answers properly refuse (mandatory), " \
              "(4) for Unsolvable: whether answers remain natural without revealing unsolvability, " \
              "(5) correctness, clarity, and style appropriateness.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(BestUserAnswerResponse) + "\n\n"
    prompt += "**CRITICAL:** For Irrelevant questions, an answer that refuses is always better. " \
              "For Unsolvable questions, don't prefer answers that reveal unsolvability. " \
              "In case of a tie, select Answer A."
    
    return prompt