from abc import ABC, abstractmethod
from contextlib import contextmanager
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
        self._initialized: bool = False
        self._pinned: bool = False

    def init(self):
        """Initialize the model. No-op if already initialized."""
        if self._initialized:
            return
        self._do_init()
        self._initialized = True

    def close(self):
        """Close the model. No-op if pinned (via keep_alive) or not initialized."""
        if not self._initialized or self._pinned:
            return
        self._do_close()
        self._initialized = False

    @contextmanager
    def keep_alive(self):
        """Context manager that keeps the model loaded across multiple operations.

        While inside the context, calls to close() are no-ops, preventing
        redundant GPU unload/reload cycles between sequential stages.
        """
        self.init()
        self._pinned = True
        try:
            yield
        finally:
            self._pinned = False
            self.close()

    @abstractmethod
    def _do_init(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def get_token_lengths(self, prompts: list[str] | list[list[dict[str, str]]]) -> list[int]:
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
    def _do_close(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def rebuild_sampling_params(self):
        """Rebuild internal sampling parameters from sampling_kwargs. Override in subclasses."""
        pass

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
