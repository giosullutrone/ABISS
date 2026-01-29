from pydantic import BaseModel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.benchmark import Conversation


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
        lines.append(f'    "{name}": "<{desc}>",')

    lines.append("}")
    return "\n".join(lines)
