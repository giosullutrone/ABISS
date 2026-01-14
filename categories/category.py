from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question
from pydantic import BaseModel


# Base categories
class Category(ABC):
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_subname() -> str | None:
        pass

    @staticmethod
    @abstractmethod
    def get_definition() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_examples() -> list[str] | None:
        pass

    @staticmethod
    @abstractmethod
    def is_solvable() -> bool:
        pass

    @staticmethod
    @abstractmethod
    def get_output() -> type[BaseModel]:
        pass

    @staticmethod
    @abstractmethod
    def get_question(db_id: str, output: BaseModel) -> list["Question"]:
        pass

    def to_dict(self) -> dict:
        return {
            "name": self.get_name(),
            "subname": self.get_subname(),
            "solvable": self.is_solvable()
        }
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Category):
            return NotImplemented
        return self.get_name() == other.get_name() and self.get_subname() == other.get_subname()