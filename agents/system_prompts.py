from db_datasets.db_dataset import DBDataset
from dataset_dataclasses.results import Conversation
from categories.category import Category
from interactions import get_conversation_history_prompt
from pydantic import BaseModel
from typing import Annotated, Optional, Literal
from pydantic import Field
from generators import model_field_descriptions


class SystemResponse(BaseModel):
    thinking_process: Annotated[str, Field(description="Step-by-step reasoning analyzing whether the question is answerable and unambiguous, or if it requires clarification. "
    "Consider: (1) whether the question can be directly translated to SQL using the schema and external knowledge, "
    "(2) whether there are ambiguities that need clarification, and (3) what specific clarification would help resolve any identified issues.")]
    response_type: Annotated[Literal["SQL", "Question"], Field(description="Either 'SQL' if the question is answerable, or 'Question' if clarification is needed.")]
    content: Annotated[str, Field(description="If response_type is 'SQL': the complete SQL query. If response_type is 'Question': the clarification question to ask the user.")]
    category: Annotated[Optional[str], Field(description="Required when response_type is 'Question': the specific ambiguity/unanswerability category name from the provided categories. Should be None for SQL responses.")]

def get_system_response_result(response: str) -> tuple[str, str, str | None]:
    response_json = SystemResponse.model_validate_json(response)
    response_type = response_json.response_type.strip().upper()
    content = response_json.content.strip()
    category = response_json.category.strip() if response_json.category else None
    return response_type, content, category

def get_interaction_prompt(db: DBDataset, conversation: Conversation, categories: list[Category]) -> str:
    categories_prompt = ""
    for category in categories:
        categories_prompt += f"**{category.get_name()}:** {category.get_definition()}\n"

    question = conversation.question

    prompt = f"You are an expert text-to-SQL system that analyzes natural language questions for database queries. " \
                "Your task is to either generate SQL queries or identify ambiguities that require clarification.\n\n"

    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, 5, None) + "\n\n"

    prompt += "## External Knowledge\n"
    prompt += f"{question.evidence if question.evidence is not None else 'N/A'}\n\n"

    prompt += "## Conversation History\n"
    prompt += get_conversation_history_prompt(conversation) + "\n\n"

    prompt += "## Question\n"
    prompt += f"{question.question}\n\n"

    prompt += "## Ambiguity and Unanswerability Categories\n"
    prompt += categories_prompt + "\n"

    prompt += "## Task\n"
    prompt += "Analyze the question and determine if it can be answered with a SQL query or if clarification is needed.\n\n"
    prompt += "**For answerable questions:** Generate a complete, syntactically correct SQL query.\n\n"
    prompt += "**For ambiguous/unanswerable questions:** Ask one focused clarification question and specify the category.\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide your response as a JSON object with:\n"
    prompt += model_field_descriptions(SystemResponse) + "\n\n"

    prompt += "Set response_type to 'SQL' for answerable questions or 'Question' for those needing clarification."
    return prompt
