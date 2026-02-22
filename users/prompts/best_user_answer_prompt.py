from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.question import QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from utils.prompt_utils import get_conversation_history_prompt
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
    prompt += "Compare the two answers for responding to a RELEVANT clarification question. "
    prompt += "Both candidates were classified as RELEVANT and attempt to disambiguate. "
    prompt += "Select the one that is more **correct** and, as a tiebreaker, more natural.\n\n"

    prompt += "**Correctness Criteria (Primary — decide the winner):**\n"
    prompt += "1. **Factual Accuracy:** Does the answer correctly convey the hidden knowledge? An answer that distorts, invents, or omits the disambiguating intent is worse, regardless of style.\n"
    prompt += "2. **Disambiguation Completeness:** Does it fully resolve the ambiguity the system asked about, or only partially?\n"
    prompt += "3. **No Information Leakage:** Does it avoid revealing SQL details (table names, column names, JOIN conditions) that a real user wouldn't know?\n\n"

    prompt += "**Style Criteria (Secondary — only used to break ties in correctness):**\n"
    prompt += "- **Naturalness:** Sounds like a real user communicating their intent\n"
    prompt += "- **Style Consistency:** Maintains the style, register, formality, and vocabulary of the original question\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide concise analysis (approximately 256 characters) comparing the answers, focusing first on correctness then on style if needed.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(BestUserAnswerRelevantResponse) + "\n\n"
    prompt += "In case of a tie, select Answer A."

    return prompt


def get_best_user_answer_technical_prompt(db: DBDataset,
                                          conversation: Conversation,
                                          generation_a: str,
                                          generation_b: str) -> str:
    """Evaluate pairs of answers to TECHNICAL clarification questions.

    NOTE: GT SQL is intentionally NOT shown here to prevent ground-truth
    leakage.  Only secondary preferences (ORDER BY, LIMIT, DISTINCT) are
    provided — the same information the answer generators received.
    """
    from users.sql_preferences import extract_secondary_preferences

    prompt = _get_best_user_answer_prompt_common(db, conversation, generation_a, generation_b, RelevancyLabel.TECHNICAL)

    question = conversation.question

    secondary = extract_secondary_preferences(question.sql)
    if secondary:
        prompt += f"\n**User's Secondary Preferences:** {secondary}\n"
    else:
        prompt += f"\n**Note:** No specific technical preferences defined — user may express uncertainty.\n"

    prompt += "\n## Candidate Answers\n"
    prompt += f"**Answer A:**\n{generation_a}\n\n"
    prompt += f"**Answer B:**\n{generation_b}\n\n"

    prompt += "## Evaluation Task\n"
    prompt += "Compare the two answers for responding to a TECHNICAL clarification question. "
    prompt += "Both candidates were classified as TECHNICAL. "
    prompt += "Select the one that is more **correct** and, as a tiebreaker, more natural.\n\n"

    prompt += "**Technical Questions Ask About:**\n"
    prompt += "- Ordering: Sort order, which field to order by (ASC/DESC)\n"
    prompt += "- Limits: How many results, top N\n"
    prompt += "- Formatting: Output format requirements\n\n"

    if secondary:
        prompt += "**Correctness Criteria (Primary — decide the winner):**\n"
        prompt += "1. **Preference Accuracy:** Does the answer correctly convey the user's secondary preferences listed above? An answer that states the wrong ordering, wrong limit, or invents preferences not listed is worse.\n"
        prompt += "2. **No Fabrication:** Does NOT add information beyond the stated preferences. Inventing extra preferences is incorrect.\n"
        prompt += "3. **No Information Leakage:** Avoids revealing SQL details (table names, column names, query structure) that a real user wouldn't know.\n\n"
    else:
        prompt += "**Correctness Criteria (Primary — decide the winner):**\n"
        prompt += "1. **Appropriate Uncertainty:** Since no preferences are defined, the answer should express genuine uncertainty ('I'm not sure', 'Either way is fine') or reasonable defaults.\n"
        prompt += "2. **No Fabrication:** Does NOT invent specific preferences that aren't defined. Stating a specific ordering/limit when none exists is incorrect.\n"
        prompt += "3. **No Information Leakage:** Avoids revealing SQL details that a real user wouldn't know.\n\n"

    prompt += "**Style Criteria (Secondary — only used to break ties in correctness):**\n"
    prompt += "- **Naturalness:** Sounds like a user stating preferences, not SQL code\n"
    prompt += "- **Style Consistency:** Maintains the style, register, formality, and vocabulary of the original question\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide concise analysis (approximately 256 characters) comparing the answers, focusing first on correctness then on style if needed.\n\n"
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

    prompt += "\n## Candidate Answers\n"
    prompt += f"**Answer A:**\n{generation_a}\n\n"
    prompt += f"**Answer B:**\n{generation_b}\n\n"

    prompt += "## Evaluation Task\n"
    prompt += "Compare the two responses to an IRRELEVANT clarification question. "
    prompt += "Both candidates were classified as IRRELEVANT. "
    prompt += "Select the one that is more **correct** and, as a tiebreaker, more natural.\n\n"

    prompt += "**Correctness Criteria (Primary — decide the winner):**\n"
    prompt += "1. **Refusal present:** The answer MUST refuse to answer the irrelevant question. An answer that engages with or attempts to answer the question is automatically worse than one that refuses.\n"
    prompt += "2. **No Information Leakage:** Does not reveal SQL details, schema information, or any data that a real user wouldn't know.\n"
    prompt += "3. **No Fabrication:** Does not invent information or answer a question the user can't answer.\n\n"

    prompt += "**Hard Decision Rules:**\n"
    prompt += "- If one refuses and the other doesn't → The refusing answer is AUTOMATICALLY better\n"
    prompt += "- If neither refuses → Both fail, but select Answer A\n\n"

    prompt += "**Style Criteria (Secondary — only when both refuse correctly):**\n"
    prompt += "- **Clarity:** How clearly does the refusal indicate it's not answering?\n"
    prompt += "- **Brevity:** Concise and direct\n"
    prompt += "- **Politeness:** Professional but firm\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide concise analysis (approximately 128 characters) focused on whether each answer properly refuses and which is more correct.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(BestUserAnswerIrrelevantResponse) + "\n\n"
    prompt += "In case of a tie, select Answer A."

    return prompt
