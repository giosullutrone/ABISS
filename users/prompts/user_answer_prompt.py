"""Stage 2: Generate natural-language-only responses grounded in matched source.

The NL-only constraint is enforced here. The model receives source material
(hidden_knowledge or SQL fragments) but must paraphrase into natural language
matching the conversation's style.
"""

from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel, Field
from typing import Annotated
from utils.prompt_utils import model_field_descriptions, get_conversation_history_prompt
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES


class UserAnswerModel(BaseModel):
    answer: Annotated[str, Field(description="Your natural language answer to the clarification question.")]


NL_CONSTRAINT = (
    "**CRITICAL CONSTRAINT:** You are a non-technical user. "
    "Respond in natural language ONLY. "
    "NEVER use SQL syntax, column names, table aliases, or query structure. "
    "Express the intent behind the information, not its SQL representation. "
    "The SQL source material is provided as reference for what you want; "
    "your answer must sound like a real non-technical user speaking in their own voice.\n\n"
)


def _style_section(conversation: Conversation) -> str:
    question_style = conversation.question.question_style
    style_desc = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    text = f"**Expected Answer Style (Match Your Original Question):**\n{style_desc}\n"
    text += "Your answer should feel like a natural continuation of your original question.\n\n"
    return text


def get_user_answer_prompt_relevant(conversation: Conversation) -> str:
    """Stage 2 prompt for RELEVANT classification. Source: hidden_knowledge."""
    question = conversation.question
    clarification = conversation.interactions[-1].system_response.system_question

    prompt = (
        "You are a user in a text-to-SQL scenario. "
        "A system asked you a clarification question that addresses the semantic ambiguity in your query. "
        "Answer it using your knowledge of what you actually want.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += f"**Clarification Question:** {clarification}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        prompt += "## Your Intent\n"
        prompt += f"**What You Actually Mean:** {question.hidden_knowledge}\n\n"

    prompt += NL_CONSTRAINT
    prompt += "## Instructions\n"
    prompt += "Use your intent to disambiguate. Be direct about which interpretation you mean. "
    prompt += "Do NOT mention SQL, databases, or technical implementation.\n\n"

    prompt += _style_section(conversation)

    prompt += "## Response Format\n"
    prompt += "Provide your answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerModel) + "\n"

    return prompt


def get_user_answer_prompt_technical(conversation: Conversation, sql_fragments: list[str]) -> str:
    """Stage 2 prompt for TECHNICAL classification. Source: matched SQL fragments."""
    question = conversation.question
    clarification = conversation.interactions[-1].system_response.system_question

    prompt = (
        "You are a user in a text-to-SQL scenario. "
        "A system asked you a technical question about implementation details. "
        "Answer it based on what you want from the results.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += f"**Clarification Question:** {clarification}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    if sql_fragments:
        prompt += "## What You Want (Reference)\n"
        prompt += "The following describes what you want from the results. "
        prompt += "Use this to inform your answer, but express it in your own words:\n"
        for frag in sql_fragments:
            prompt += f"- {frag}\n"
        prompt += "\n"
    else:
        prompt += "## What You Want (Reference)\n"
        prompt += "No specific preferences are defined for this aspect of the query.\n\n"

    prompt += NL_CONSTRAINT
    prompt += "## Instructions\n"
    if sql_fragments:
        prompt += "Express the intent behind the reference information in natural language. "
        prompt += "For example, 'ORDER BY count DESC' should become something like "
        prompt += "'I want the most popular ones first' (colloquial) or "
        prompt += "'Sort by frequency in descending order' (formal). "
        prompt += "Match the style of the original question.\n\n"
    else:
        prompt += "You don't have a specific preference for what the system is asking about. "
        prompt += "Say so directly rather than being vague. "
        prompt += "Examples: 'I don't have a preference for the ordering, just show me the data', "
        prompt += "'I don't really care about that, just pick whatever makes sense', "
        prompt += "'That detail doesn't matter to me, focus on the main results'.\n\n"

    prompt += _style_section(conversation)

    prompt += "## Response Format\n"
    prompt += "Provide your answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerModel) + "\n"

    return prompt


def get_user_answer_prompt_irrelevant(conversation: Conversation) -> str:
    """Stage 2 prompt for IRRELEVANT classification. No source, refuse."""
    question = conversation.question
    clarification = conversation.interactions[-1].system_response.system_question

    prompt = (
        "You are a user in a text-to-SQL scenario. "
        "A system asked you an irrelevant clarification question. "
        "Politely but firmly refuse to answer.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    prompt += f"**Clarification Question:** {clarification}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    prompt += "## Instructions\n"
    prompt += "Politely but firmly refuse to answer. "
    prompt += "Examples: 'That's not relevant to my question', 'I can't answer that', "
    prompt += "'Could you focus on my original question instead?'\n\n"

    prompt += _style_section(conversation)

    prompt += "## Response Format\n"
    prompt += "Provide your answer as a JSON object with:\n"
    prompt += model_field_descriptions(UserAnswerModel) + "\n"

    return prompt
