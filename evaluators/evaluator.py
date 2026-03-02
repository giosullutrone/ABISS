from abc import ABC, abstractmethod
from dataset_dataclasses.benchmark import Conversation
from typing import Any


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, conversations: list[Conversation]) -> Any:
        pass
