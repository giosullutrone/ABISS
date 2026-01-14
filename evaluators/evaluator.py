from abc import ABC, abstractmethod
from dataset_dataclasses.results import Conversation


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, conversations: list[Conversation]) -> list[Conversation]:
        pass
