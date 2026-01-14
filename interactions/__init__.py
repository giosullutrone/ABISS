from enum import Enum
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.results import Conversation
from db_datasets.db_dataset import DBDataset


class RelevancyLabel(str, Enum):
    RELEVANT = "Relevant"
    TECHNICAL = "Technical"
    IRRELEVANT = "Irrelevant"

class UserAnswerStyle(Enum):
    PRECISE = "precise"
    CONVERSATIONAL = "conversational"

class UserKnowledgeLevel(Enum):
    FULL = "full"
    NL = "nl"
    NONE = "none"

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

def get_db_knowledge_level_prompt(db: DBDataset, user_knowledge_level: UserKnowledgeLevel, db_descriptions: dict[str, str] | None, conversation: "Conversation") -> str:
    if user_knowledge_level == UserKnowledgeLevel.FULL:
        db_schema = db.get_schema_prompt(conversation.question.db_id, rows=5, db_sql_manipulation=None)
        return f"- The full database schema:\n{db_schema}\n"
    elif user_knowledge_level == UserKnowledgeLevel.NL:
        db_description = db_descriptions.get(conversation.question.db_id) # type: ignore
        return f"- A description of the database schema: '{db_description}'\n"
    else:
        return ""