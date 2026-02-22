from abc import ABC, abstractmethod
from dataset_dataclasses.benchmark import Conversation, SystemResponse
from dataset_dataclasses.question import Question
from categories.category import Category


class System(ABC):
    def __init__(self, agent_name: str, categories: list[Category], max_steps: int) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.categories = categories
        self.max_steps = max_steps

    def get_category(self, questions: list[Question]) -> list[Category] | None:
        """
        Classify each question into a category.
        Returns None if this system does not support classification,
        in which case only NO_CATEGORY mode will be used.
        """
        return None

    @abstractmethod
    def get_system_response(
        self,
        conversations: list[Conversation],
        categories_to_use: list[Category | None],
        current_steps: list[int]
    ) -> list[SystemResponse]:
        pass
