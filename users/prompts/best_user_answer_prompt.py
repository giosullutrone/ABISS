from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.question import QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from utils.prompt_utils import get_conversation_history_prompt
from utils.knowledge_level_utils import KNOWLEDGE_LEVEL_INFO
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES
from pydantic import BaseModel
from typing import Annotated, Literal
from pydantic import Field
from utils.prompt_utils import model_field_descriptions


# Response models for each relevancy type
class BestUserAnswerRelevantResponse(BaseModel):
    answer: Annotated[Literal["A", "B"], Field(description="Final selection: 'A' if Answer A is better, 'B' if Answer B is better. Put only 'A' or 'B'.")]


class BestUserAnswerTechnicalResponse(BaseModel):
    answer: Annotated[Literal["A", "B"], Field(description="Final selection: 'A' if Answer A is better, 'B' if Answer B is better. Put only 'A' or 'B'.")]


class BestUserAnswerIrrelevantResponse(BaseModel):
    answer: Annotated[Literal["A", "B"], Field(description="Final selection: 'A' if Answer A is better, 'B' if Answer B is better. Put only 'A' or 'B'.")]


# Result extractors
def get_best_user_answer_relevant_result(response: BaseModel) -> int:
    """Parses the model response and returns '0' or '1' based on whether Answer A is better or B."""
    answer = BestUserAnswerRelevantResponse.model_validate(response).answer.strip().upper()
    if "A" in answer:
        return 0
    elif "B" in answer:
        return 1
    raise ValueError("Invalid answer in BestUserAnswerRelevantResponse: must contain 'A' or 'B'.")


def get_best_user_answer_technical_result(response: BaseModel) -> int:
    """Parses the model response and returns '0' or '1' based on whether Answer A is better or B."""
    answer = BestUserAnswerTechnicalResponse.model_validate(response).answer.strip().upper()
    if "A" in answer:
        return 0
    elif "B" in answer:
        return 1
    raise ValueError("Invalid answer in BestUserAnswerTechnicalResponse: must contain 'A' or 'B'.")


def get_best_user_answer_irrelevant_result(response: BaseModel) -> int:
    """Parses the model response and returns '0' or '1' based on whether Answer A is better or B."""
    answer = BestUserAnswerIrrelevantResponse.model_validate(response).answer.strip().upper()
    if "A" in answer:
        return 0
    elif "B" in answer:
        return 1
    raise ValueError("Invalid answer in BestUserAnswerIrrelevantResponse: must contain 'A' or 'B'.")


def _get_best_user_answer_prompt_common(db: DBDataset, 
                                        conversation: Conversation, 
                                        generation_a: str, 
                                        generation_b: str,
                                        relevancy_type: RelevancyLabel) -> str:
    """Shared prompt components for all best user answer evaluation."""
    prompt = f"You are an expert evaluator for user answers to {relevancy_type.value} clarification questions in text-to-SQL scenarios.\n\n"
    
    prompt += "## Context\n"
    user_knowledge_level = conversation.user_knowledge_level
    prompt += KNOWLEDGE_LEVEL_INFO[user_knowledge_level]['description']
    prompt += get_conversation_history_prompt(conversation)
    
    question = conversation.question
    question_style = question.question_style
    style_description = STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES[question_style]
    prompt += f"**Expected Answer Style:**\n{style_description}\n"
    
    return prompt


def get_best_user_answer_relevant_prompt(db: DBDataset, 
                                         conversation: Conversation, 
                                         generation_a: str, 
                                         generation_b: str) -> str:
    """Evaluate pairs of answers to RELEVANT clarification questions.
    Note: Only solvable questions can be labeled RELEVANT (answerable questions can only be TECHNICAL or IRRELEVANT).
    """
    prompt = _get_best_user_answer_prompt_common(db, conversation, generation_a, generation_b, RelevancyLabel.RELEVANT)
    
    question = conversation.question
    
    # Only solvable questions can reach here - answerable questions can't be RELEVANT
    prompt += f"\n**Question Type:** Solvable - requires disambiguation\n"
    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Disambiguating Intent):** {question.hidden_knowledge}\n"
    
    prompt += "\n## Candidate Answers\n"
    prompt += f"**Answer A:**\n{generation_a}\n\n"
    prompt += f"**Answer B:**\n{generation_b}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Compare the two answers for responding to a RELEVANT clarification question:\n\n"
    
    prompt += "**Evaluation Guidelines:**\n"
    prompt += "- Answers should use the provided hidden knowledge to disambiguate\n"
    prompt += "- Select the answer that better clarifies the user's intent\n"
    prompt += "- Prefer answers that directly address the ambiguity\n"
    prompt += "- The answer should help the system understand which interpretation is correct\n\n"
    
    prompt += "**Evaluation Criteria:**\n"
    prompt += "- **Clarity:** How well does the answer resolve the ambiguity?\n"
    prompt += "- **Use of Information:** Effective use of hidden knowledge\n"
    prompt += "- **Knowledge Level Appropriateness:** Uses language matching user's schema knowledge (technical/domain/intuitive)\n"
    prompt += "- **Naturalness:** Sounds like a real user communicating their intent\n"
    prompt += "- **Style Consistency:** Maintains the style, register, formality, and vocabulary of the original question\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide concise analysis (approximately 256 characters) comparing the answers based on how well they use hidden knowledge to disambiguate.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(BestUserAnswerRelevantResponse) + "\n\n"
    prompt += "In case of a tie, select Answer A."
    
    return prompt


def get_best_user_answer_technical_prompt(db: DBDataset, 
                                          conversation: Conversation, 
                                          generation_a: str, 
                                          generation_b: str) -> str:
    """Evaluate pairs of answers to TECHNICAL clarification questions."""
    prompt = _get_best_user_answer_prompt_common(db, conversation, generation_a, generation_b, RelevancyLabel.TECHNICAL)
    
    question = conversation.question
    
    if question.sql:
        prompt += f"\n**User's Output Preferences (encoded in SQL):** {question.sql}\n"
        prompt += f"**Note:** Technical preferences about ordering, limits, formatting extractable from this SQL.\n"
    else:
        prompt += f"\n**Note:** No specific technical preferences defined - user may express uncertainty.\n"
    
    prompt += "\n## Candidate Answers\n"
    prompt += f"**Answer A:**\n{generation_a}\n\n"
    prompt += f"**Answer B:**\n{generation_b}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Compare the two answers for responding to a TECHNICAL clarification question:\n\n"
    
    prompt += "**Technical Questions Ask About:**\n"
    prompt += "- Ordering: Sort order, which field to order by (ASC/DESC)\n"
    prompt += "- Limits: How many results, top N\n"
    prompt += "- Formatting: Output format requirements\n"
    prompt += "- Aggregation: How to compute averages, sums, counts\n"
    prompt += "- Filtering: Specific threshold or condition details\n\n"
    
    if question.sql:
        prompt += "**Evaluation for Questions with Defined Preferences:**\n"
        prompt += "- Answers should extract the relevant preference from the SQL\n"
        prompt += "- Prefer answers that accurately convey the preference naturally\n"
        prompt += "- Answers should sound like a user stating what they want (not that they 'know SQL')\n"
        prompt += "- Do NOT prefer answers that add information not in the SQL\n\n"
    else:
        prompt += "**Evaluation for Questions without Defined Preferences:**\n"
        prompt += "- Answers should express uncertainty appropriately\n"
        prompt += "- Prefer natural uncertainty: 'I'm not sure', 'Either way is fine'\n"
        prompt += "- Or reasonable defaults if the user would have a preference\n\n"
    
    prompt += "**Evaluation Criteria:**\n"
    prompt += "- **Accuracy:** Correctly extracts preference from SQL (if available)\n"
    prompt += "- **Knowledge Level Appropriateness:** Uses language matching user's schema knowledge (technical/natural/vague)\n"
    prompt += "- **Naturalness:** Sounds like a user stating preferences, not SQL code\n"
    prompt += "- **Appropriateness:** Uncertain when preferences undefined, specific when defined\n"
    prompt += "- **Information Boundary:** Doesn't add information not present in SQL\n"
    prompt += "- **Style Consistency:** Maintains the style, register, formality, and vocabulary of the original question\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide concise analysis (approximately 256 characters) comparing how well each answer conveys technical preferences from the SQL or expresses appropriate uncertainty.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(BestUserAnswerTechnicalResponse) + "\n\n"
    prompt += "In case of a tie, select Answer A."
    
    return prompt


def get_best_user_answer_irrelevant_prompt(db: DBDataset, 
                                           conversation: Conversation, 
                                           generation_a: str, 
                                           generation_b: str) -> str:
    """Evaluate pairs of responses to IRRELEVANT clarification questions."""
    prompt = _get_best_user_answer_prompt_common(db, conversation, generation_a, generation_b, RelevancyLabel.IRRELEVANT)
    
    prompt += "\n**Note:** For irrelevant questions, refusal takes priority over style - clarity is key.\n\n"
    
    prompt += "## Candidate Answers\n"
    prompt += f"**Answer A:**\n{generation_a}\n\n"
    prompt += f"**Answer B:**\n{generation_b}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Compare the two responses to an IRRELEVANT clarification question.\n\n"
    
    prompt += "**CRITICAL REQUIREMENT:**\n"
    prompt += "Both answers MUST refuse to answer the irrelevant question.\n\n"
    
    prompt += "**Decision Rules:**\n"
    prompt += "1. If one answer refuses and the other doesn't → The refusing answer is AUTOMATICALLY better\n"
    prompt += "2. If both refuse → Select the clearer, more natural refusal\n"
    prompt += "3. If neither refuses → Both fail, but select Answer A\n\n"
    
    prompt += "**Good Refusal Examples:**\n"
    prompt += "- 'I can't answer that'\n"
    prompt += "- 'That's not relevant to my question'\n"
    prompt += "- 'I'm not sure how that helps'\n"
    prompt += "- 'That's not what I'm asking about'\n\n"
    
    prompt += "**Bad Responses (Not Refusals):**\n"
    prompt += "- Attempting to answer the irrelevant question\n"
    prompt += "- Providing information unrelated to the original query\n"
    prompt += "- Engaging with the irrelevant topic\n\n"
    
    prompt += "**Evaluation Criteria (when both refuse):**\n"
    prompt += "- **Clarity:** How clearly does the refusal indicate it's not answering?\n"
    prompt += "- **Politeness:** Professional but firm\n"
    prompt += "- **Relevance Signal:** Clearly indicates the question isn't relevant\n"
    prompt += "- **Brevity:** Concise and direct\n"
    prompt += "- **Style Consistency:** When possible, maintains the original question's register (but refusal clarity is more important)\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide concise analysis (approximately 128 characters) focused on whether each answer properly refuses and which refusal is clearer.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(BestUserAnswerIrrelevantResponse) + "\n\n"
    prompt += "**REMINDER:** If one refuses and the other doesn't, the refusal is automatically better. In case of a tie, select Answer A."
    
    return prompt
