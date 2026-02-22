"""Unified prompt that produces both a relevancy label and a user answer in a
single model call.  This eliminates ground-truth SQL leakage by never exposing
the full GT SQL to the user simulator — only programmatically extracted
secondary preferences (ORDER BY, LIMIT, DISTINCT) are provided.
"""

from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from utils.prompt_utils import model_field_descriptions, get_conversation_history_prompt
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES
from users.sql_preferences import extract_secondary_preferences


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class UserResponseSolvableModel(BaseModel):
    """For solvable (ambiguous) questions — all three relevancy labels valid."""
    relevancy: Annotated[
        Literal["Relevant", "Technical", "Irrelevant"],
        Field(description=(
            "Classification of the clarification question: "
            "'Relevant' if it addresses the semantic ambiguity, "
            "'Technical' if it asks about SQL implementation details (ordering, limits, columns), "
            "or 'Irrelevant' if it doesn't help or tries to extract SQL. "
            "Put only 'Relevant', 'Technical', or 'Irrelevant'."
        )),
    ]
    answer: Annotated[str, Field(description="Your answer to the clarification question.")]


class UserResponseAnswerableModel(BaseModel):
    """For answerable questions — only Technical or Irrelevant."""
    relevancy: Annotated[
        Literal["Technical", "Irrelevant"],
        Field(description=(
            "Classification of the clarification question: "
            "'Technical' if it asks about SQL implementation details (ordering, limits, columns), "
            "or 'Irrelevant' if it doesn't help or tries to extract SQL. "
            "Put only 'Technical' or 'Irrelevant'."
        )),
    ]
    answer: Annotated[str, Field(description="Your answer to the clarification question.")]


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------

def get_user_response_solvable_result(response: BaseModel) -> tuple[RelevancyLabel, str]:
    validated = UserResponseSolvableModel.model_validate(response)
    label_str = validated.relevancy.strip().lower()
    if "irrelevant" in label_str:
        label = RelevancyLabel.IRRELEVANT
    elif "relevant" in label_str:
        label = RelevancyLabel.RELEVANT
    elif "technical" in label_str:
        label = RelevancyLabel.TECHNICAL
    else:
        raise ValueError(f"Invalid relevancy in UserResponseSolvableModel: {validated.relevancy}")
    return label, validated.answer.strip()


def get_user_response_answerable_result(response: BaseModel) -> tuple[RelevancyLabel, str]:
    validated = UserResponseAnswerableModel.model_validate(response)
    label_str = validated.relevancy.strip().lower()
    if "technical" in label_str:
        label = RelevancyLabel.TECHNICAL
    elif "irrelevant" in label_str:
        label = RelevancyLabel.IRRELEVANT
    else:
        raise ValueError(f"Invalid relevancy in UserResponseAnswerableModel: {validated.relevancy}")
    return label, validated.answer.strip()


# ---------------------------------------------------------------------------
# Shared definitions
# ---------------------------------------------------------------------------

def _relevancy_definitions(include_relevant: bool) -> str:
    text = ""
    idx = 1
    if include_relevant:
        text += (
            f"**{idx}. RELEVANT** (Addresses semantic ambiguity):\n"
            "- Directly addresses the semantic ambiguity using your hidden knowledge\n"
            "- Helps clarify which interpretation you intend\n"
            "- Focuses on natural language disambiguation (not SQL implementation)\n"
            "- NOT about which columns/tables to use — about WHICH MEANING is intended\n"
            "- Examples: 'Which date do you mean — enrollment or graduation?', 'Do you want student count or course count?'\n\n"
        )
        idx += 1

    text += (
        f"**{idx}. TECHNICAL** (Asks about SQL implementation details):\n"
        "- Focuses on output preferences and implementation: ordering, limits, formatting\n"
        "- Asks about which columns, tables, or fields to use\n"
        "- Can be answered from your secondary preferences or with reasonable defaults\n"
        "- NOT about semantic meaning — about HOW to implement or present results\n"
        "- Examples: 'Order by newest first?', 'Limit to top 10?'\n\n"
    )
    idx += 1

    text += (
        f"**{idx}. IRRELEVANT** (Doesn't help resolve the query):\n"
        "- Doesn't address the ambiguity or provide useful technical details\n"
        "- Tries to extract the SQL solution directly (CHEATING)\n"
        "- Completely off-topic or tangential to the query\n"
        "- Examples: 'What's the SQL?', 'Can you write the query for me?'\n\n"
    )
    return text


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def get_user_response_prompt_solvable(conversation: Conversation) -> str:
    """For solvable (ambiguous) questions — RELEVANT, TECHNICAL, or IRRELEVANT."""
    question = conversation.question
    assert conversation.interactions[-1].system_response.system_question is not None

    prompt = (
        "You are an expert user simulator for text-to-SQL scenarios. "
        "A text-to-SQL system has asked you a clarification question about your original query. "
        "You must first classify the clarification question, then provide an appropriate answer.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += f"**Question Type:** Solvable — has ambiguity that can be resolved through clarification\n"

    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Your Disambiguating Intent):** {question.hidden_knowledge}\n"

    secondary = extract_secondary_preferences(question.sql)
    if secondary:
        prompt += f"**Your Secondary Preferences:** {secondary}\n"
    else:
        prompt += "**Your Secondary Preferences:** None defined — express uncertainty for technical details.\n"

    prompt += f"\n**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    prompt += "## Relevancy Definitions\n"
    prompt += _relevancy_definitions(include_relevant=True)

    prompt += "## How to Answer Based on Classification\n"
    prompt += (
        "- **If RELEVANT:** Use the hidden knowledge to disambiguate. "
        "Be direct about which interpretation you mean.\n"
        "- **If TECHNICAL:** Use your secondary preferences if available, "
        "otherwise express uncertainty ('Either way is fine', 'I'm not sure').\n"
        "- **If IRRELEVANT:** Politely but firmly refuse to answer. "
        "Examples: 'That's not relevant to my question', 'I can't answer that'.\n\n"
    )

    prompt += "**CRITICAL RULES:**\n"
    prompt += "- RELEVANT = addresses semantic ambiguity (which meaning?) | TECHNICAL = asks implementation (columns, ordering, limits) | IRRELEVANT = doesn't help\n"
    prompt += "- Questions asking about columns/tables to use are TECHNICAL (not Relevant)\n"
    prompt += "- Questions trying to extract SQL directly are ALWAYS IRRELEVANT\n\n"

    question_style = question.question_style
    style_desc = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    prompt += f"**Expected Answer Style (Match Your Original Question):**\n{style_desc}\n"
    prompt += "Your answer should feel like a natural continuation of your original question.\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide brief reasoning (approximately 256 characters), then a JSON object with:\n"
    prompt += model_field_descriptions(UserResponseSolvableModel) + "\n"

    return prompt


def get_user_response_prompt_answerable(conversation: Conversation) -> str:
    """For answerable questions — only TECHNICAL or IRRELEVANT."""
    question = conversation.question
    assert conversation.interactions[-1].system_response.system_question is not None

    prompt = (
        "You are an expert user simulator for text-to-SQL scenarios. "
        "A text-to-SQL system has asked you a clarification question about your original query. "
        "You must first classify the clarification question, then provide an appropriate answer.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += f"**Question Type:** Answerable — question is already clear, doesn't need semantic clarification\n"

    secondary = extract_secondary_preferences(question.sql)
    if secondary:
        prompt += f"**Your Secondary Preferences:** {secondary}\n"
    else:
        prompt += "**Your Secondary Preferences:** None defined — express uncertainty for technical details.\n"

    prompt += f"\n**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    prompt += "## Relevancy Definitions\n"
    prompt += _relevancy_definitions(include_relevant=False)

    prompt += "## How to Answer Based on Classification\n"
    prompt += (
        "- **If TECHNICAL:** Use your secondary preferences if available, "
        "otherwise express uncertainty ('Either way is fine', 'I'm not sure').\n"
        "- **If IRRELEVANT:** Politely but firmly refuse to answer. "
        "Examples: 'That's not relevant to my question', 'I can't answer that'.\n\n"
    )

    prompt += "**CRITICAL RULES:**\n"
    prompt += "- Questions asking about columns/tables are TECHNICAL\n"
    prompt += "- Questions trying to extract SQL directly are ALWAYS IRRELEVANT\n"
    prompt += "- If clarification asks about semantics: classify as IRRELEVANT (question is already clear)\n\n"

    question_style = question.question_style
    style_desc = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    prompt += f"**Expected Answer Style (Match Your Original Question):**\n{style_desc}\n"
    prompt += "Your answer should feel like a natural continuation of your original question.\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide brief reasoning (approximately 256 characters), then a JSON object with:\n"
    prompt += model_field_descriptions(UserResponseAnswerableModel) + "\n"

    return prompt
