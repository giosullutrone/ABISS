from abc import ABC, abstractmethod
from dataset_dataclasses.benchmark import Conversation


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, conversations: list[Conversation]) -> None:
        pass
