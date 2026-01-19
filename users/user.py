from models.model import Model
from users.question_relevancy import QuestionRelevancy
from users.user_answer import UserAnswer
from users.schema_to_nl import SchemaToNL
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

        self.question_relevancy_interaction = QuestionRelevancy(self.db, self.models)

        self.schema_to_nl_interaction = SchemaToNL(self.db, self.models)
        db_descriptions: dict[str, str] | None = None

        db_descriptions = self.schema_to_nl_interaction.generate_descriptions(db_ids)

        self.user_answer_interaction = UserAnswer(self.db, self.models, db_descriptions)
    
    def get_relevancy(self, conversations: list[Conversation]) -> None:       
        self.question_relevancy_interaction.get_relevancy(conversations)

    def get_answers(self, conversations: list[Conversation]) -> None:
        self.user_answer_interaction.get_user_answers(conversations)
