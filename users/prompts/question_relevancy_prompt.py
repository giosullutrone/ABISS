from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from utils.prompt_utils import model_field_descriptions
from dataset_dataclasses.benchmark import RelevancyLabel
from typing import Literal


class QuestionRelevancyResponse(BaseModel):
    answer: Annotated[Literal["Relevant", "Technical", "Irrelevant"], Field(description="Final classification: 'Relevant' if the clarification question helps disambiguate the original question using the hidden knowledge, "
    "'Technical' if it focuses on SQL technical aspects unrelated to the ambiguity, or 'Irrelevant' if it doesn't help with disambiguation or tries to extract hidden information. Put only 'Relevant', 'Technical', or 'Irrelevant'.")]

def get_question_relevancy_result(response: BaseModel) -> RelevancyLabel:
    answer = QuestionRelevancyResponse.model_validate(response).answer.strip().lower()
    if 'relevant' in answer:
        return RelevancyLabel.RELEVANT
    elif 'technical' in answer:
        return RelevancyLabel.TECHNICAL
    elif 'irrelevant' in answer:
        return RelevancyLabel.IRRELEVANT
    raise ValueError("Invalid answer in QuestionRelevancyResponse: must contain 'Relevant', 'Technical', or 'Irrelevant'.")

def get_relevancy_prompt(conversation: Conversation) -> str:
    prompt = "You are an expert evaluator for text-to-SQL clarification question relevance. " \
             "Your task is to assess whether a clarification question helps disambiguate an ambiguous user query.\n\n"
    
    prompt += "## Context\n"
    prompt += f"**Original Ambiguous Question:** {conversation.question.question}\n"

    if conversation.question.evidence:
        prompt += f"**Additional Context:** {conversation.question.evidence}\n"
    
    # Determine question type
    assert isinstance(conversation.question, QuestionUnanswerable), "Question must be of type QuestionUnanswerable."
    is_answerable = conversation.question.category.is_answerable()
    is_solvable = conversation.question.category.is_solvable()
    
    # Show hidden knowledge only for solvable questions (for unsolvable, hidden_knowledge is the feedback we want the system to provide)
    if conversation.question.hidden_knowledge and is_solvable:
        prompt += f"**Hidden Knowledge (Disambiguating Information):** {conversation.question.hidden_knowledge}\n"
    
    if conversation.question.sql:
        prompt += f"**Ground Truth SQL:** {conversation.question.sql}\n"
    
    # Indicate question type
    if is_answerable:
        prompt += f"**Question Type:** Answerable (no disambiguation needed)\n"
    elif is_solvable:
        prompt += f"**Question Type:** Unanswerable but Solvable (requires disambiguation)\n"
    else:
        prompt += f"**Question Type:** Unsolvable (cannot be answered - missing external knowledge, improper question, or missing schema elements)\n"
    
    prompt += f"**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Determine if the clarification question is relevant based on the question type.\n\n"
    
    prompt += "**Classification Categories:**\n\n"
    
    prompt += "**For Answerable or Unanswerable-Solvable questions:**\n"
    prompt += "- **Relevant:** The clarification question directly addresses the ambiguity and helps apply the hidden knowledge " \
              "to resolve it. Focuses on natural language disambiguation rather than technical SQL details.\n\n"
    
    prompt += "**For Unsolvable questions:**\n"
    prompt += "- **Relevant:** The clarification question attempts to identify WHY the question is unsolvable " \
              "(e.g., asking about missing external knowledge, schema limitations, or whether the question is proper). " \
              "These questions help the system provide appropriate feedback about unsolvability.\n\n"
    
    prompt += "**For all question types:**\n"
    prompt += "- **Technical:** The question focuses on technical SQL aspects (ordering, limits, formatting) " \
              "unrelated to the core ambiguity or unsolvability. Can be answered from the ground truth SQL query.\n\n"
    
    prompt += "- **Irrelevant:** The question (1) does not help resolve the ambiguity or identify unsolvability, " \
              "(2) tries to extract the SQL solution directly (cheating), " \
              "(3) is completely unrelated to disambiguation, technical aspects, or identifying the problem, or " \
              "(4) asks tangential/off-topic information. Questions attempting to extract the SQL directly are Irrelevant.\n\n"
    
    prompt += "**CRITICAL:** Questions attempting to extract the SQL solution directly are cheating the benchmark " \
              "and MUST be classified as Irrelevant, not Technical. Questions that are simply unrelated to both " \
              "the ambiguity and technical SQL aspects are also Irrelevant.\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a step-by-step analysis of whether the clarification question is relevant based on the question type. " \
              "Your reasoning should be concise but thorough (approximately 512 characters), addressing: " \
              "(1) the question type (answerable/solvable/unsolvable), " \
              "(2) for solvable: whether it addresses the ambiguity and hidden knowledge, " \
              "(3) for unsolvable: whether it helps identify why the question cannot be solved, " \
              "(4) whether it focuses on disambiguation/problem identification rather than extracting SQL, and " \
              "(5) if technical, whether it can be answered from the ground truth SQL.\n\n"
    prompt += "Then provide your final classification as a JSON object with:\n"
    prompt += model_field_descriptions(QuestionRelevancyResponse) + "\n\n"
    prompt += "Choose exactly one classification: 'Relevant', 'Technical', or 'Irrelevant'. " \
              "Remember: questions trying to extract SQL directly are 'Irrelevant', not 'Technical'."
    
    return prompt
