from db_datasets.db_dataset import DBDataset
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from utils.prompt_utils import model_field_descriptions, get_conversation_history_prompt
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES
from utils.knowledge_level_utils import KNOWLEDGE_LEVEL_INFO


# Response models for each relevancy type
class UserAnswerRelevantResponse(BaseModel):
    answer: Annotated[str, Field(description="The final user answer to the relevant clarification question, using hidden knowledge to disambiguate the semantic ambiguity.")]


class UserAnswerTechnicalResponse(BaseModel):
    answer: Annotated[str, Field(description="The final user answer to the technical clarification question, providing implementation details based on your output preferences.")]


class UserAnswerIrrelevantResponse(BaseModel):
    answer: Annotated[str, Field(description="The final user response refusing to answer the irrelevant clarification question.")]


# Result extractors
def get_user_answer_relevant_result(response: BaseModel) -> str:
    return UserAnswerRelevantResponse.model_validate(response).answer.strip()


def get_user_answer_technical_result(response: BaseModel) -> str:
    return UserAnswerTechnicalResponse.model_validate(response).answer.strip()


def get_user_answer_irrelevant_result(response: BaseModel) -> str:
    return UserAnswerIrrelevantResponse.model_validate(response).answer.strip()


def _get_user_answer_prompt_common(db: DBDataset, 
                                   conversation: Conversation, 
                                   relevancy_type: RelevancyLabel) -> str:
    """Shared prompt components for all user answer generation."""
    question = conversation.question
    assert conversation.interactions[-1].system_response.system_question is not None, "Last system response must be a question."

    prompt = f"You are an expert user simulator for text-to-SQL scenarios. " \
             f"Your task is to answer a {relevancy_type.value} clarification question"
    
    if relevancy_type == RelevancyLabel.RELEVANT:
        prompt += " that addresses semantic ambiguity in your original question.\n\n"
    elif relevancy_type == RelevancyLabel.TECHNICAL:
        prompt += " about implementation details and output preferences.\n\n"
    else:  # IRRELEVANT
        prompt += " that doesn't help with your original question.\n\n"

    prompt += "## Context\n"
    user_knowledge_level = conversation.user_knowledge_level
    prompt += KNOWLEDGE_LEVEL_INFO[user_knowledge_level]['description']

    prompt += f"\n**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    
    return prompt


def get_user_answer_relevant_prompt(db: DBDataset, 
                                    conversation: Conversation) -> str:
    """Generate a prompt for answering RELEVANT clarification questions.
    Note: Only solvable questions can be labeled RELEVANT (answerable questions can only be TECHNICAL or IRRELEVANT).
    """
    prompt = _get_user_answer_prompt_common(db, conversation, RelevancyLabel.RELEVANT)
    
    question = conversation.question
    
    # Only solvable questions can reach here - answerable questions can't be RELEVANT
    prompt += f"**Question Type:** Solvable - requires disambiguation through interaction\n"
    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Your Disambiguating Intent):** {question.hidden_knowledge}\n"
    
    prompt += f"**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    prompt += "## Task\n"
    prompt += "Answer the clarification question using your hidden knowledge to disambiguate and resolve the semantic ambiguity.\n\n"
    prompt += "**Guidelines:**\n"
    
    user_knowledge_level = conversation.user_knowledge_level
    prompt += KNOWLEDGE_LEVEL_INFO[user_knowledge_level]['relevant_guidelines'] + "\n"
    
    prompt += "- Use the provided hidden knowledge to clarify your intent\n"
    prompt += "- Be direct and clear about which interpretation you mean\n"
    prompt += "- Help the system understand the specific meaning you intended\n"
    prompt += "- Do NOT answer questions about which columns/tables to use - those are TECHNICAL, not RELEVANT\n"
    prompt += "- CRITICALLY: Match the style and register of your original question\n\n"
    
    question_style = question.question_style
    style_description = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    prompt += f"**Expected Answer Style (Match Your Original Question):**\n{style_description}\n"
    prompt += "Your answer should feel like a natural continuation of your original question - maintain consistency in formality, vocabulary, and tone.\n\n"
    
    prompt += KNOWLEDGE_LEVEL_INFO[user_knowledge_level]['style_fusion'] + "\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide brief reasoning (approximately 256 characters) about how to use the hidden knowledge to answer the clarification question.\n\n"
    prompt += "Then provide your final answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerRelevantResponse) + "\n"

    return prompt


def get_user_answer_technical_prompt(db: DBDataset, 
                                     conversation: Conversation) -> str:
    """Generate a prompt for answering TECHNICAL clarification questions."""
    prompt = _get_user_answer_prompt_common(db, conversation, RelevancyLabel.TECHNICAL)
    
    question = conversation.question
    
    if question.sql:
        prompt += f"**Your Output Preferences (encoded in SQL):** {question.sql}\n"
        prompt += f"**Note:** You don't think in SQL, but you have preferences about columns, tables, ordering, limits, filtering criteria, and formatting. " \
                  f"Extract these preferences from the SQL to answer technical questions.\n"
    else:
        prompt += f"**Note:** No specific technical preferences are defined. Express uncertainty or provide reasonable defaults.\n"
    
    prompt += f"**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    prompt += "## Task\n"
    prompt += "Answer the technical question based on your output preferences.\n\n"
    
    prompt += "**Technical Question Types:**\n"
    prompt += "- **Columns/Tables:** Which specific columns or tables to use for the query?\n"
    prompt += "- **Ordering:** Do you want results sorted? In what order? (ASC/DESC, by which field?)\n"
    prompt += "- **Limits:** How many results? Top N? All results?\n"
    prompt += "- **Formatting:** Specific output format requirements?\n"
    prompt += "- **Aggregation:** How to compute averages, sums, counts?\n"
    prompt += "- **Filtering/Criteria:** Specific threshold, time range, or condition details (e.g., last year, above $1000, only active records)?\n\n"
    
    prompt += "**Guidelines:**\n"
    
    user_knowledge_level = conversation.user_knowledge_level
    if question.sql:
        prompt += "- Extract the relevant preference from your SQL\n"
        prompt += KNOWLEDGE_LEVEL_INFO[user_knowledge_level]['technical_guidelines'] + "\n"
    else:
        prompt += "- Express uncertainty: 'I'm not sure', 'Either way is fine', 'Whatever is standard'\n"
        prompt += "- Provide reasonable defaults if you have a preference\n"
    
    prompt += "- Do NOT make up information not present in your preferences\n"
    prompt += "- CRITICALLY: Match the style and register of your original question\n\n"
    
    question_style = question.question_style
    style_description = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    prompt += f"**Expected Answer Style (Match Your Original Question):**\n{style_description}\n"
    prompt += "Your answer should feel like a natural continuation of your original question - maintain consistency in formality, vocabulary, and tone.\n\n"
    
    prompt += "**IMPORTANT - Fusing Knowledge Level with Style:**\n"
    prompt += KNOWLEDGE_LEVEL_INFO[user_knowledge_level]['technical_style_fusion'] + "\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide brief reasoning (approximately 256 characters) about what technical preference to extract or that you're uncertain.\n\n"
    prompt += "Then provide your final answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerTechnicalResponse) + "\n"

    return prompt


def get_user_answer_irrelevant_prompt(db: DBDataset, 
                                      conversation: Conversation) -> str:
    """Generate a prompt for responding to IRRELEVANT clarification questions."""
    prompt = _get_user_answer_prompt_common(db, conversation, RelevancyLabel.IRRELEVANT)
    
    question = conversation.question
    
    prompt += f"**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    prompt += "## Task\n"
    prompt += "This clarification question is irrelevant - it doesn't help resolve your original question. You must refuse to answer it.\n\n"
    
    prompt += "**Why Questions Are Irrelevant:**\n"
    prompt += "- Asks for information unrelated to the query\n"
    prompt += "- Attempts to extract the SQL solution directly (cheating)\n"
    prompt += "- Completely off-topic or tangential\n"
    prompt += "- Doesn't address the ambiguity or provide useful technical details\n\n"
    
    prompt += "**Guidelines:**\n"
    prompt += "- Politely but firmly refuse to answer\n"
    prompt += "- Indicate it's not relevant to your question\n"
    prompt += "- Keep response brief and clear\n"
    prompt += "- Examples:\n"
    prompt += "  - 'I can't answer that'\n"
    prompt += "  - 'That's not relevant to my question'\n"
    prompt += "  - 'I'm not sure how that helps'\n"
    prompt += "  - 'That's not what I'm asking about'\n"
    prompt += "- CRITICALLY: While refusing, try to match the style and register of your original question\n\n"
    
    question_style = question.question_style
    style_description = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    prompt += f"**Expected Answer Style (adapted for refusal):**\n{style_description}\n"
    prompt += "However, for irrelevant questions, clarity of refusal takes priority over strict style matching.\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide brief reasoning (approximately 128 characters) about why you're refusing.\n\n"
    prompt += "Then provide your final answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerIrrelevantResponse) + "\n"

    return prompt