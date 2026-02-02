from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from utils.prompt_utils import model_field_descriptions
from dataset_dataclasses.benchmark import RelevancyLabel
from typing import Literal


def _get_relevant_definition() -> str:
    """Definition of RELEVANT classification - only applies to solvable questions."""
    return (
        "**RELEVANT** (Addresses semantic ambiguity):\n"
        "- ✅ Valid ONLY for Solvable questions\n"
        "- Directly addresses the semantic ambiguity using hidden knowledge\n"
        "- Helps clarify which interpretation the user intends\n"
        "- Focuses on natural language disambiguation (not SQL implementation)\n"
        "- NOT about which columns/tables to use - about WHICH MEANING is intended\n"
        "- Examples: 'Which date do you mean - enrollment or graduation?', 'Do you want student count or course count?'\n"
    )


def _get_technical_definition() -> str:
    """Definition of TECHNICAL classification - applies to both answerable and solvable questions."""
    return (
        "**TECHNICAL** (Asks about SQL implementation details):\n"
        "- ✅ Valid for both Answerable and Solvable questions\n"
        "- Focuses on output preferences and implementation: ordering, limits, formatting, aggregation, filtering criteria\n"
        "- Asks about which columns, tables, or fields to use\n"
        "- Asks about limiting/filtering results by specific criteria (time ranges, thresholds, conditions)\n"
        "- Can be answered from the ground truth SQL or database schema\n"
        "- NOT about semantic meaning - about HOW to implement or present results\n"
        "- Examples: 'Order by newest first?', 'Limit to top 10?', 'Use the enrollment_date column?', 'Join with students table?', 'Limit to last year?', 'Only include results above $1000?'\n"
    )


def _get_irrelevant_definition() -> str:
    """Definition of IRRELEVANT classification - applies to both answerable and solvable questions."""
    return (
        "**IRRELEVANT** (Doesn't help resolve the query):\n"
        "- ✅ Valid for both Answerable and Solvable questions\n"
        "- Doesn't address the ambiguity (if solvable) or provide useful technical details\n"
        "- Tries to extract the SQL solution directly (CHEATING)\n"
        "- Completely off-topic or tangential to the query\n"
        "- Examples: 'What's the SQL?', 'Unrelated topic', 'Can you write the query for me?'\n"
    )


# Answerable questions - only TECHNICAL or IRRELEVANT
class QuestionRelevancyAnswerableResponse(BaseModel):
    answer: Annotated[Literal["Technical", "Irrelevant"], Field(description="Final classification for answerable questions: "
    "'Technical' if it focuses on SQL implementation details (ordering, limits, formatting, columns, tables), or 'Irrelevant' if it doesn't help or tries to extract SQL. "
    "Note: 'Relevant' is NOT valid for answerable questions - they're already clear. Put only 'Technical' or 'Irrelevant'.")]


# Solvable questions - all three labels possible
class QuestionRelevancySolvableResponse(BaseModel):
    answer: Annotated[Literal["Relevant", "Technical", "Irrelevant"], Field(description="Final classification for solvable questions: "
    "'Relevant' if it addresses semantic ambiguity using hidden knowledge (NOT about columns/tables), 'Technical' if it focuses on SQL implementation details (columns, tables, ordering, limits), "
    "or 'Irrelevant' if it doesn't help with disambiguation or tries to extract SQL. Put only 'Relevant', 'Technical', or 'Irrelevant'.")]


def get_question_relevancy_answerable_result(response: BaseModel) -> RelevancyLabel:
    answer = QuestionRelevancyAnswerableResponse.model_validate(response).answer.strip().lower()
    if 'technical' in answer:
        return RelevancyLabel.TECHNICAL
    elif 'irrelevant' in answer:
        return RelevancyLabel.IRRELEVANT
    raise ValueError("Invalid answer in QuestionRelevancyAnswerableResponse: must contain 'Technical' or 'Irrelevant'.")


def get_question_relevancy_solvable_result(response: BaseModel) -> RelevancyLabel:
    answer = QuestionRelevancySolvableResponse.model_validate(response).answer.strip().lower()
    if 'relevant' in answer:
        return RelevancyLabel.RELEVANT
    elif 'technical' in answer:
        return RelevancyLabel.TECHNICAL
    elif 'irrelevant' in answer:
        return RelevancyLabel.IRRELEVANT
    raise ValueError("Invalid answer in QuestionRelevancySolvableResponse: must contain 'Relevant', 'Technical', or 'Irrelevant'.")


def _get_relevancy_prompt_common(conversation: Conversation) -> str:
    """Shared prompt components for both answerable and solvable questions."""
    prompt = "You are an expert evaluator for text-to-SQL clarification question relevance. " \
             "Your task is to classify whether a clarification question is Relevant, Technical, or Irrelevant.\n\n"
    
    prompt += "## Context\n"
    prompt += f"**Original Question:** {conversation.question.question}\n"

    if conversation.question.evidence:
        prompt += f"**Additional Context:** {conversation.question.evidence}\n"
    
    return prompt

def get_relevancy_prompt_answerable(conversation: Conversation) -> str:
    """Prompt for classifying clarification questions on answerable questions (only TECHNICAL or IRRELEVANT)."""
    prompt = _get_relevancy_prompt_common(conversation)
    
    prompt += f"**Question Type:** Answerable - question is already clear, doesn't need semantic clarification\n"
    if conversation.question.sql:
        prompt += f"**Ground Truth SQL:** {conversation.question.sql}\n"
    
    prompt += f"**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    
    prompt += "## Classification Task\n"
    prompt += "Classify the clarification question as Technical or Irrelevant.\n\n"
    
    prompt += "**Classification Definitions:**\n\n"
    
    # Use shared definition functions - numbering adjusted for answerable (no RELEVANT)
    prompt += "**1. " + _get_technical_definition().replace("**TECHNICAL**", "TECHNICAL**", 1)
    prompt += "\n"
    prompt += "**2. " + _get_irrelevant_definition().replace("**IRRELEVANT**", "IRRELEVANT**", 1)
    prompt += "\n"
    
    prompt += "**CRITICAL RULES:**\n"
    prompt += "- Questions asking about columns/tables to use are TECHNICAL\n"
    prompt += "- Questions trying to extract SQL directly are ALWAYS IRRELEVANT (not Technical)\n"
    prompt += "- If clarification asks about semantics: classify as IRRELEVANT (question is already clear)\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide concise reasoning (approximately 256 characters) addressing: " \
              "(1) what the clarification question asks about (technical vs. other), " \
              "(2) your classification choice.\n\n"
    prompt += "Then provide your final classification as a JSON object with:\n"
    prompt += model_field_descriptions(QuestionRelevancyAnswerableResponse) + "\n"
    
    return prompt


def get_relevancy_prompt_solvable(conversation: Conversation) -> str:
    """Prompt for classifying clarification questions on solvable questions (RELEVANT, TECHNICAL, or IRRELEVANT)."""
    prompt = _get_relevancy_prompt_common(conversation)
    
    prompt += f"**Question Type:** Solvable - has ambiguity that can be resolved through clarification\n"
    if isinstance(conversation.question, QuestionUnanswerable) and conversation.question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Disambiguating Information):** {conversation.question.hidden_knowledge}\n"
    if conversation.question.sql:
        prompt += f"**Ground Truth SQL:** {conversation.question.sql}\n"
    
    prompt += f"**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    
    prompt += "## Classification Task\n"
    prompt += "Classify the clarification question as Relevant, Technical, or Irrelevant.\n\n"
    
    prompt += "**Classification Definitions:**\n\n"
    
    # Use shared definition functions
    prompt += "**1. " + _get_relevant_definition().replace("**RELEVANT**", "RELEVANT**", 1)
    prompt += "\n"
    prompt += "**2. " + _get_technical_definition().replace("**TECHNICAL**", "TECHNICAL**", 1)
    prompt += "\n"
    prompt += "**3. " + _get_irrelevant_definition().replace("**IRRELEVANT**", "IRRELEVANT**", 1)
    prompt += "\n"
    
    prompt += "**CRITICAL RULES:**\n"
    prompt += "- RELEVANT = addresses semantic ambiguity (which meaning?) | TECHNICAL = asks implementation (which column/table, how to order, etc.) | IRRELEVANT = doesn't help\n"
    prompt += "- Questions asking about columns/tables to use are TECHNICAL (not Relevant)\n"
    prompt += "- Questions trying to extract SQL directly are ALWAYS IRRELEVANT (not Technical)\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide concise reasoning (approximately 256 characters) addressing: " \
              "(1) what the clarification question asks about (semantics vs. technical vs. other), " \
              "(2) your classification choice.\n\n"
    prompt += "Then provide your final classification as a JSON object with:\n"
    prompt += model_field_descriptions(QuestionRelevancySolvableResponse) + "\n"
    
    return prompt