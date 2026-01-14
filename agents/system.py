from abc import ABC, abstractmethod
from dataset_dataclasses.results import Conversation
from dataset_dataclasses.system import SystemResponseQuestion, SystemResponseSQL


class System(ABC):
    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name

    @abstractmethod
    def get_system_responses(self, conversations: list[Conversation]) -> list[SystemResponseQuestion | SystemResponseSQL]:
        raise NotImplementedError("Subsequent interactions are not yet implemented for SystemLLM.")
