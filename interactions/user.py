from models.model import Model
from interactions.question_relevancy import QuestionRelevancy
from interactions.user_answer import UserAnswer
from interactions.schema_to_nl import SchemaToNL
from db_datasets.db_dataset import DBDataset
from dataset_dataclasses.results import Conversation
from interactions import UserKnowledgeLevel, UserAnswerStyle


class User:
    def __init__(self, 
                 agent_name: str, 
                 db: DBDataset, 
                 models: list[Model], 
                 db_ids: list[str], 
                 user_knowledge_level: UserKnowledgeLevel, 
                 user_answer_style: UserAnswerStyle) -> None:
        super().__init__()
        self.agent_name: str = agent_name
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.db_ids: list[str] = db_ids

        self.question_relevancy_interaction = QuestionRelevancy(self.db, self.models)

        self.schema_to_nl_interaction = SchemaToNL(self.db, self.models)
        db_descriptions: dict[str, str] | None = None

        if user_knowledge_level == UserKnowledgeLevel.NL:
            db_descriptions = self.schema_to_nl_interaction.generate_descriptions(db_ids)

        self.user_answer_interaction = UserAnswer(self.db, self.models, user_knowledge_level, user_answer_style, db_descriptions)
    
    def get_relevancy(self, conversations: list[Conversation]) -> list[Conversation]:       
        return self.question_relevancy_interaction.get_relevancy(conversations)

    def get_answers(self, conversations: list[Conversation]) -> list[Conversation]:
        return self.user_answer_interaction.get_user_answers(conversations)
