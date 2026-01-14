from abc import ABC, abstractmethod
from dataset_dataclasses.question import QuestionUnanswerable


class Validator(ABC):
    @abstractmethod
    def validate(self, questions: list[QuestionUnanswerable]) -> list[bool]:
        pass