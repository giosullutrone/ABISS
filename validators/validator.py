from abc import ABC, abstractmethod
from dataset_dataclasses.question import Question
from dataset_dataclasses.council_tracking import ValidationStageResult


class Validator(ABC):
    @abstractmethod
    def validate(self, questions: list[Question]) -> ValidationStageResult:
        pass
