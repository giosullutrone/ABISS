from pydantic import BaseModel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.benchmark import Conversation


def get_conversation_history_prompt(conversation: "Conversation") -> str:
    # Only include completed interactions (both system and user responses present).
    # The current/in-progress interaction is excluded since its user_response is None.
    completed = [t for t in conversation.interactions if t.user_response is not None]
    if not completed:
        return ""
    question = conversation.question
    prompt = "The full conversation history is as follows:\n"
    prompt += f"User: {question.question}\n"
    if question.evidence:
        prompt += f"Evidence: {question.evidence}\n"
    for turn in completed:
        prompt += f"System: {turn.system_response}\n"
        prompt += f"User: {turn.user_response}\n"
    prompt += "\n"
    return prompt

def model_field_descriptions(model: type[BaseModel]) -> str:
    lines = ["{"]

    for name, field in model.model_fields.items():
        assert field.description is not None, f"Field {name} is missing description."
        desc = field.description
        lines.append(f'    "{name}": "<{desc}>",')

    lines.append("}")
    return "\n".join(lines)
