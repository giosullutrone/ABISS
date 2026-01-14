from abc import ABC, abstractmethod


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
    def generate_batch_with_constraints(self, prompts: list[str] | list[list[dict[str, str]]], constraints: dict) -> list[str]:
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def close(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

