from abc import ABC, abstractmethod
from dataset_dataclasses.benchmark import Conversation, SystemResponse
from categories.category import Category


class System(ABC):
    def __init__(self, agent_name: str, categories: list[Category], max_steps: int) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.categories = categories
        self.max_steps = max_steps

    @abstractmethod
    def get_category(self, conversations: list[Conversation]) -> list[Category]:
        pass

    @abstractmethod
    def get_system_response(
        self, 
        conversations: list[Conversation], 
        categories_to_use: list[Category | None],
        current_steps: list[int]
    ) -> list[SystemResponse]:
        pass
