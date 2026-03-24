"""Stage 1: Classify clarification questions and locate relevant SQL AST nodes.

The classifier sees raw SQL fragments in the AST node list to enable
semantic matching. The NL-only constraint is applied in Stage 2.
"""

from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from utils.prompt_utils import model_field_descriptions, get_conversation_history_prompt
from users.sql_ast import SQLNode, format_nodes_for_prompt


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UserClassifySolvable(BaseModel):
    """For solvable (ambiguous) questions -- all three relevancy labels valid."""
    relevancy: Annotated[
        Literal["Relevant", "Technical", "Irrelevant"],
        Field(description=(
            "Classification of the clarification question: "
            "'Relevant' if it addresses the semantic ambiguity, "
            "'Technical' if it asks about SQL implementation details, "
            "or 'Irrelevant' if it doesn't help or tries to extract SQL. "
            "Put only 'Relevant', 'Technical', or 'Irrelevant'."
        )),
    ]
    node_ids: Annotated[
        list[int],
        Field(description=(
            "If Technical, list ALL SQL node IDs the question relates to. "
            "If Relevant or Irrelevant, leave this as an empty list []."
        )),
    ]


class UserClassifyAnswerable(BaseModel):
    """For answerable questions -- only Technical or Irrelevant."""
    relevancy: Annotated[
        Literal["Technical", "Irrelevant"],
        Field(description=(
            "Classification of the clarification question: "
            "'Technical' if it asks about SQL implementation details, "
            "or 'Irrelevant' if it doesn't help or tries to extract SQL. "
            "Put only 'Technical' or 'Irrelevant'."
        )),
    ]
    node_ids: Annotated[
        list[int],
        Field(description=(
            "If Technical, list ALL SQL node IDs the question relates to. "
            "If Irrelevant, leave this as an empty list []."
        )),
    ]


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def get_classify_solvable_result(response: BaseModel) -> tuple[RelevancyLabel, list[int]]:
    validated = UserClassifySolvable.model_validate(response)
    label_str = validated.relevancy.strip().lower()
    if "irrelevant" in label_str:
        label = RelevancyLabel.IRRELEVANT
    elif "relevant" in label_str:
        label = RelevancyLabel.RELEVANT
    elif "technical" in label_str:
        label = RelevancyLabel.TECHNICAL
    else:
        raise ValueError(f"Invalid relevancy: {validated.relevancy}")
    return label, validated.node_ids


def get_classify_answerable_result(response: BaseModel) -> tuple[RelevancyLabel, list[int]]:
    validated = UserClassifyAnswerable.model_validate(response)
    label_str = validated.relevancy.strip().lower()
    if "technical" in label_str:
        label = RelevancyLabel.TECHNICAL
    elif "irrelevant" in label_str:
        label = RelevancyLabel.IRRELEVANT
    else:
        raise ValueError(f"Invalid relevancy: {validated.relevancy}")
    return label, validated.node_ids


# ---------------------------------------------------------------------------
# Relevancy definitions (shared)
# ---------------------------------------------------------------------------

def _relevancy_definitions(include_relevant: bool) -> str:
    text = ""
    idx = 1
    if include_relevant:
        text += (
            f"**{idx}. RELEVANT** (Addresses semantic ambiguity):\n"
            "- Directly addresses the semantic ambiguity using hidden knowledge\n"
            "- Helps clarify which interpretation the user intends\n"
            "- Focuses on natural language disambiguation (not SQL implementation)\n"
            "- NOT about which columns/tables to use -- about WHICH MEANING is intended\n\n"
        )
        idx += 1

    text += (
        f"**{idx}. TECHNICAL** (Asks about SQL implementation details):\n"
        "- Focuses on output preferences and implementation: ordering, limits, grouping, aggregation\n"
        "- Asks about which columns, tables, or fields to use\n"
        "- Can be answered using the SQL reference nodes listed above\n"
        "- NOT about semantic meaning -- about HOW to implement or present results\n\n"
    )
    idx += 1

    text += (
        f"**{idx}. IRRELEVANT** (Doesn't help resolve the query):\n"
        "- Doesn't address the ambiguity or provide useful technical details\n"
        "- Tries to extract the SQL solution directly\n"
        "- Completely off-topic or tangential to the query\n\n"
    )
    return text


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def get_user_classify_prompt_solvable(conversation: Conversation, nodes: list[SQLNode]) -> str:
    """Stage 1 prompt for solvable (ambiguous) questions."""
    question = conversation.question
    assert conversation.interactions[-1].system_response.system_question is not None

    prompt = (
        "You are a classification engine for a text-to-SQL user simulator. "
        "A text-to-SQL system has asked a clarification question. "
        "Classify it and identify which SQL nodes it relates to.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += "**Question Type:** Solvable -- has ambiguity that can be resolved through clarification\n"

    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Disambiguating Intent):** {question.hidden_knowledge}\n"

    prompt += f"\n**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    nodes_text = format_nodes_for_prompt(nodes)
    if nodes_text:
        prompt += "## SQL Reference Nodes\n"
        prompt += nodes_text + "\n\n"
    else:
        prompt += "## SQL Reference Nodes\nNo SQL nodes available.\n\n"

    prompt += "## Relevancy Definitions\n"
    prompt += _relevancy_definitions(include_relevant=True)

    prompt += "## Classification Rules\n"
    prompt += "- RELEVANT = addresses semantic ambiguity (which meaning?)\n"
    prompt += "- TECHNICAL = asks implementation details (columns, ordering, limits, grouping, aggregation)\n"
    prompt += "- IRRELEVANT = doesn't help or tries to extract SQL\n"
    prompt += "- Questions asking about columns/tables to use are TECHNICAL (not Relevant)\n"
    prompt += "- If TECHNICAL, list ALL node IDs the question relates to\n"
    prompt += "- If RELEVANT or IRRELEVANT, leave node_ids as []\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide brief reasoning (approximately 128 characters), then a JSON object with:\n"
    prompt += model_field_descriptions(UserClassifySolvable) + "\n"

    return prompt


def get_user_classify_prompt_answerable(conversation: Conversation, nodes: list[SQLNode]) -> str:
    """Stage 1 prompt for answerable questions (no semantic ambiguity)."""
    question = conversation.question
    assert conversation.interactions[-1].system_response.system_question is not None

    prompt = (
        "You are a classification engine for a text-to-SQL user simulator. "
        "A text-to-SQL system has asked a clarification question about an already-clear query. "
        "Classify it and identify which SQL nodes it relates to.\n\n"
    )

    prompt += "## Context\n"
    prompt += f"**Original Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Additional Context:** {question.evidence}\n"
    prompt += "**Question Type:** Answerable -- question is already clear, no semantic ambiguity\n"

    prompt += f"\n**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    prompt += get_conversation_history_prompt(conversation)

    nodes_text = format_nodes_for_prompt(nodes)
    if nodes_text:
        prompt += "## SQL Reference Nodes\n"
        prompt += nodes_text + "\n\n"
    else:
        prompt += "## SQL Reference Nodes\nNo SQL nodes available.\n\n"

    prompt += "## Relevancy Definitions\n"
    prompt += _relevancy_definitions(include_relevant=False)

    prompt += "## Classification Rules\n"
    prompt += "- TECHNICAL = asks implementation details (columns, ordering, limits, grouping, aggregation)\n"
    prompt += "- IRRELEVANT = doesn't help or tries to extract SQL\n"
    prompt += "- Semantic clarification questions are IRRELEVANT (question is already clear)\n"
    prompt += "- If TECHNICAL, list ALL node IDs the question relates to\n"
    prompt += "- If IRRELEVANT, leave node_ids as []\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide brief reasoning (approximately 128 characters), then a JSON object with:\n"
    prompt += model_field_descriptions(UserClassifyAnswerable) + "\n"

    return prompt
