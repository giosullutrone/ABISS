from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from utils.prompt_utils import model_field_descriptions
from dataset_dataclasses.benchmark import RelevancyLabel
from typing import Literal


# Answerable questions - only TECHNICAL or IRRELEVANT
class QuestionRelevancyAnswerableResponse(BaseModel):
    answer: Annotated[Literal["Technical", "Irrelevant"], Field(description="Final classification for answerable questions: "
    "'Technical' if it focuses on SQL implementation details (ordering, limits, formatting), or 'Irrelevant' if it doesn't help or tries to extract SQL. "
    "Note: 'Relevant' is NOT valid for answerable questions - they're already clear. Put only 'Technical' or 'Irrelevant'.")]


# Solvable questions - all three labels possible
class QuestionRelevancySolvableResponse(BaseModel):
    answer: Annotated[Literal["Relevant", "Technical", "Irrelevant"], Field(description="Final classification for solvable questions: "
    "'Relevant' if it addresses semantic ambiguity using hidden knowledge, 'Technical' if it focuses on SQL implementation details, "
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
    
    prompt += "**1. TECHNICAL** (Asks about SQL implementation details):\n"
    prompt += "- ✅ Valid for Answerable questions\n"
    prompt += "- Focuses on output preferences and implementation: ordering, limits, formatting, aggregation\n"
    prompt += "- Can be answered from the ground truth SQL\n"
    prompt += "- NOT about semantic meaning - about HOW to present results\n"
    prompt += "- Examples: 'Order by newest first?', 'Limit to top 10?', 'Include duplicates?'\n\n"
    
    prompt += "**2. IRRELEVANT** (Doesn't help resolve the query):\n"
    prompt += "- ✅ Valid for Answerable questions\n"
    prompt += "- Doesn't provide useful technical details\n"
    prompt += "- Tries to extract the SQL solution directly (CHEATING)\n"
    prompt += "- Completely off-topic or tangential to the query\n"
    prompt += "- Attempts semantic clarification (question is already clear!)\n"
    prompt += "- Examples: 'What's the SQL?', 'Unrelated topic', 'Which tables should I join?'\n\n"
    
    prompt += "**CRITICAL RULES:**\n"
    prompt += "- This is an ANSWERABLE question → CANNOT be classified as RELEVANT\n"
    prompt += "- If the clarification asks about semantics: classify as IRRELEVANT (question is already clear)\n"
    prompt += "- Only TECHNICAL (asking implementation details) or IRRELEVANT are valid\n"
    prompt += "- Questions trying to extract SQL directly are ALWAYS IRRELEVANT (not Technical)\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide concise reasoning (approximately 256 characters) addressing: " \
              "(1) this is answerable so only Technical/Irrelevant are valid, " \
              "(2) what the clarification question asks about (technical vs. other), " \
              "(3) your classification choice.\n\n"
    prompt += "Then provide your final classification as a JSON object with:\n"
    prompt += model_field_descriptions(QuestionRelevancyAnswerableResponse) + "\n\n"
    prompt += "**REMINDER:** Answerable questions CANNOT be RELEVANT. Only TECHNICAL or IRRELEVANT are valid."
    
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
    
    prompt += "**1. RELEVANT** (Addresses semantic ambiguity):\n"
    prompt += "- ✅ Valid for Solvable questions\n"
    prompt += "- Directly addresses the semantic ambiguity using hidden knowledge\n"
    prompt += "- Helps clarify which interpretation the user intends\n"
    prompt += "- Focuses on natural language disambiguation (not SQL implementation)\n"
    prompt += "- Examples: 'Which date do you mean?', 'Do you want students or courses?'\n\n"
    
    prompt += "**2. TECHNICAL** (Asks about SQL implementation details):\n"
    prompt += "- ✅ Valid for Solvable questions\n"
    prompt += "- Focuses on output preferences and implementation: ordering, limits, formatting, aggregation\n"
    prompt += "- Can be answered from the ground truth SQL\n"
    prompt += "- NOT about semantic meaning - about HOW to present results\n"
    prompt += "- Examples: 'Order by newest first?', 'Limit to top 10?', 'Include duplicates?'\n\n"
    
    prompt += "**3. IRRELEVANT** (Doesn't help resolve the query):\n"
    prompt += "- ✅ Valid for Solvable questions\n"
    prompt += "- Doesn't address the ambiguity or provide useful technical details\n"
    prompt += "- Tries to extract the SQL solution directly (CHEATING)\n"
    prompt += "- Completely off-topic or tangential to the query\n"
    prompt += "- Examples: 'What's the SQL?', 'Unrelated topic', 'Which tables should I join?'\n\n"
    
    prompt += "**CRITICAL RULES:**\n"
    prompt += "- This is a SOLVABLE question → RELEVANT, TECHNICAL, or IRRELEVANT are all possible\n"
    prompt += "- RELEVANT = addresses the ambiguity | TECHNICAL = asks implementation | IRRELEVANT = doesn't help\n"
    prompt += "- Questions trying to extract SQL directly are ALWAYS IRRELEVANT (not Technical)\n"
    prompt += "- Distinguish: semantic disambiguation (Relevant) vs. implementation details (Technical)\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide concise reasoning (approximately 256 characters) addressing: " \
              "(1) this is solvable so all three labels are possible, " \
              "(2) what the clarification question asks about (semantics vs. technical vs. other), " \
              "(3) your classification choice.\n\n"
    prompt += "Then provide your final classification as a JSON object with:\n"
    prompt += model_field_descriptions(QuestionRelevancySolvableResponse) + "\n\n"
    prompt += "**REMINDER:** Choose RELEVANT (semantic), TECHNICAL (implementation), or IRRELEVANT (doesn't help)."
    
    return prompt

