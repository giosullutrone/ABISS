from abc import ABC, abstractmethod
from dataset_dataclasses.question import Question


class Validator(ABC):
    @abstractmethod
    def validate(self, questions: list[Question]) -> list[bool]:
        pass