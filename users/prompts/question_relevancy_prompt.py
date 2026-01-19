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
    
    assert isinstance(conversation.question, QuestionUnanswerable), "Question must be of type QuestionUnanswerable."

    if conversation.question.hidden_knowledge:
        prompt += f"**Hidden Knowledge (Disambiguating Information):** {conversation.question.hidden_knowledge}\n"
    
    if conversation.question.sql:
        prompt += f"**Ground Truth SQL:** {conversation.question.sql}\n"
    
    prompt += f"**Clarification Question:** {conversation.interactions[-1].system_response.system_question}\n\n"
    
    prompt += "## Evaluation Task\n"
    prompt += "Determine if the clarification question is relevant to disambiguating the original question using the hidden knowledge.\n\n"
    
    prompt += "**Classification Categories:**\n\n"
    
    prompt += "- **Relevant:** The clarification question directly addresses the ambiguity in the original question " \
              "and helps apply the hidden knowledge to resolve it. It focuses on natural language disambiguation " \
              "rather than technical SQL details.\n\n"
    
    prompt += "- **Technical:** The clarification question is valid but focuses on technical SQL aspects " \
              "(such as ordering, limits, formatting, or output preferences) that are unrelated to the core ambiguity. " \
              "These questions can be answered using information extractable from the ground truth SQL query.\n\n"
    
    prompt += "- **Irrelevant:** The clarification question falls into one of these categories: " \
              "(1) does not help resolve the ambiguity, " \
              "(2) tries to extract the hidden knowledge or SQL solution directly (cheating the benchmark), " \
              "(3) is completely unrelated to the disambiguation task, technical details, or the problem in the question, or " \
              "(4) asks about tangential or off-topic information. Questions that attempt to directly extract the SQL query " \
              "or its core logic must be classified as Irrelevant.\n\n"
    
    prompt += "**CRITICAL:** Questions attempting to extract the SQL solution directly are cheating the benchmark " \
              "and MUST be classified as Irrelevant, not Technical. Questions that are simply unrelated to both " \
              "the ambiguity and technical SQL aspects are also Irrelevant.\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a step-by-step analysis of whether the clarification question is relevant to the original ambiguous question and hidden knowledge. " \
              "Your reasoning should be concise but thorough (approximately 512 characters), addressing: " \
              "(1) whether the clarification question addresses the ambiguity in the original question, " \
              "(2) whether it relates to the hidden knowledge that disambiguates the question, " \
              "(3) whether it focuses on disambiguation rather than extracting hidden information or the SQL solution, and " \
              "(4) if technical, whether it can be answered from the ground truth SQL.\n\n"
    prompt += "Then provide your final classification as a JSON object with:\n"
    prompt += model_field_descriptions(QuestionRelevancyResponse) + "\n\n"
    prompt += "Choose exactly one classification: 'Relevant', 'Technical', or 'Irrelevant'. " \
              "Remember: questions trying to extract SQL directly are 'Irrelevant', not 'Technical'."
    
    return prompt
