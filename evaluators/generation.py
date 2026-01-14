from evaluators.evaluator import Evaluator
from dataset_dataclasses.results import Conversation
from dataset_dataclasses.system import SystemResponseSQL
from dataset_dataclasses.question import QuestionUnanswerable
from db_datasets.db_dataset import DBDataset


class Generation(Evaluator):
    def __init__(self, db: DBDataset) -> None:
        self.db: DBDataset = db

    def evaluate(self, conversations: list[Conversation]) -> list[Conversation]:
        questions = [conversation.question for conversation in conversations]
        system_responses = [conversation.interactions[-1].system_response for conversation in conversations]
        generations: list[bool | None] = [None] * len(questions)
        for idx, (question, system_response) in enumerate(zip(questions, system_responses)):
            # Generation is only applicable for unanswerable questions with SQL responses
            if isinstance(system_response, SystemResponseSQL) and isinstance(question, QuestionUnanswerable) and question.category.is_solvable():
                # We check if the generated SQL resolves the ambiguity (i.e. they are equivalent)
                assert question.sql is not None, "Ground truth SQL should be available for unanswerable questions."
                generations[idx] = self.db.compare_query_results(question.db_id, system_response.sql, question.sql, None)
        
        for conversation, generation in zip(conversations, generations):
            conversation.interactions[-1].solved = generation
        return conversations
