from pydantic import BaseModel
from dataset_dataclasses.benchmark import UserKnowledgeLevel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.benchmark import Conversation
    from db_datasets.db_dataset import DBDataset


def get_conversation_history_prompt(conversation: "Conversation") -> str:
    prompt = ""
    if len(conversation.interactions) > 1: # The first turn is the original question
        prompt += f"The full conversation history is as follows:\n"
        prompt += f"{conversation.question.question}\n"
        for turn in conversation.interactions:
            prompt += f"- System: {turn.system_response}\n"
            prompt += f"- User: {turn.user_response}\n"
        prompt += f"\n"
    return prompt

def model_field_descriptions(model: type[BaseModel]) -> str:
    lines = ["{"]

    for name, field in model.model_fields.items():
        assert field.description is not None, f"Field {name} is missing description."
        desc = field.description
        lines.append(f'    "{name}": "{desc}",')

    lines.append("}")
    return "\n".join(lines)

def get_db_knowledge_level_prompt(db: "DBDataset", user_knowledge_level: UserKnowledgeLevel, db_descriptions: dict[str, str] | None, conversation: "Conversation") -> str:
    if user_knowledge_level == UserKnowledgeLevel.FULL:
        db_schema = db.get_schema_prompt(conversation.question.db_id, rows=5)
        return f"- The full database schema:\n{db_schema}\n"
    elif user_knowledge_level == UserKnowledgeLevel.NL:
        db_description = db_descriptions.get(conversation.question.db_id) # type: ignore
        return f"- A description of the database schema: '{db_description}'\n"
    else:
        return ""