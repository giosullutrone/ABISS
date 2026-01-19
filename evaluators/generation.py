from evaluators.evaluator import Evaluator
from dataset_dataclasses.benchmark import Conversation
from db_datasets.db_dataset import DBDataset


class Generation(Evaluator):
    def __init__(self, db: DBDataset) -> None:
        self.db: DBDataset = db

    def evaluate(self, conversations: list[Conversation]) -> None:
        """
        Set the solved to True if the predicted SQL matches the ground truth SQL
        using relaxed semantic equivalence (ignores row order, allows column supersets).
        """
        for conversation in conversations:
            sql = conversation.question.sql
            predicted_sql = conversation.predicted_sql
            
            if predicted_sql is None or sql is None:
                conversation.solved = False
                continue
            
            # Use relaxed semantic equivalence comparison
            result = self.db.compare_query_results(
                db_id=conversation.question.db_id,
                sql_query_1=predicted_sql,  # generated query
                sql_query_2=sql  # ground truth query
            )
            
            conversation.solved = result