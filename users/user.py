from models.model import Model
from users.user_response import UserResponse
from db_datasets.db_dataset import DBDataset
from dataset_dataclasses.benchmark import Conversation


class User:
    def __init__(self,
                 agent_name: str,
                 db: DBDataset,
                 models: list[Model],
                 db_ids: list[str]) -> None:
        super().__init__()
        self.agent_name: str = agent_name
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.db_ids: list[str] = db_ids

        self.user_response_interaction = UserResponse(self.db, self.models)

    def get_response(self, conversations: list[Conversation]) -> None:
        """Unified step: classifies relevancy AND generates user answer."""
        self.user_response_interaction.get_response(conversations)
