from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import cast


class Model(ABC):
    def __init__(self, 
                 model_name: str, 
                 lora_name: str | None = None,
                 system_prompt: str | None = None,
                 sampling_kwargs: dict | None = None,
                 model_kwargs: dict | None = None,
                 max_batch_with_text_size: int = 10000):
        self.model_name: str = model_name
        self.lora_name: str | None = lora_name
        self.system_prompt: str | None = system_prompt
        self.sampling_kwargs: dict | None = sampling_kwargs
        self.model_kwargs: dict | None = model_kwargs
        self.max_batch_with_text_size: int = max_batch_with_text_size

    @abstractmethod
    def init(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def generate_batch(self, prompts: list[str] | list[list[dict[str, str]]]) -> list[str]:
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def generate_batch_with_constraints(self, prompts: list[str] | list[list[dict[str, str]]], constraints: list[type[BaseModel]]) -> list[BaseModel]:
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def generate_batch_with_constraints_unsafe(self, prompts: list[str] | list[list[dict[str, str]]], constraints: list[type[BaseModel]]) -> list[BaseModel | None]:
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def close(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def convert_prompt_to_conversation_if_needed(self, prompts: list[str] | list[list[dict[str, str]]]) -> list[list[dict[str, str]]]:
        """
        Converts a single string prompt into a conversation format expected by the model if needed.
        """
        if not isinstance(prompts[0], str) and isinstance(prompts[0], list):
            return cast(list[list[dict[str, str]]], prompts)
        
        prompts = cast(list[str], prompts)
        conversations: list[list[dict[str, str]]] = []
        for prompt in prompts:
            conversation: list[dict[str, str]] = []
            if self.system_prompt is not None:
                conversation.append({"role": "system", "content": self.system_prompt})
            conversation.append({"role": "user", "content": prompt})
            conversations.append(conversation)
        return conversations