from evaluators.evaluator import Evaluator
from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import Question, QuestionUnanswerable
from db_datasets.db_dataset import DBDataset


class Generation(Evaluator):
    def __init__(self, db: DBDataset) -> None:
        self.db: DBDataset = db

    def evaluate(self, conversations: list[Conversation]) -> None:
        """
        Set the solved to True if the predicted SQL matches the ground truth SQL.
        """
        sqls = [conversation.question.sql for conversation in conversations]
        predicted_sqls = [conversation.interactions[-1].system_response.system_sql for conversation in conversations]

        results = [(sql == psql) if psql is not None else False for sql, psql in zip(sqls, predicted_sqls)]

        for conversation, result in zip(conversations, results):
            conversation.solved = result