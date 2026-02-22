from evaluators.evaluator import Evaluator
from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from db_datasets.db_dataset import DBDataset


class Generation(Evaluator):
    def __init__(self, db: DBDataset) -> None:
        self.db: DBDataset = db

    def evaluate(self, conversations: list[Conversation]) -> None:
        """
        Set the solved flag based on SQL correctness.

        Sets solved to None if:
        - The question is unanswerable (unsolvable) and the system did not produce SQL
          (this is the expected behavior — feedback is the correct response)

        Sets solved to False if:
        - The system did not produce SQL for an answerable/solvable question
        - The question has no ground truth SQL
        - The predicted SQL does not match the ground truth SQL
        - The system produced SQL for an unanswerable question (wrong action)
        """
        for conversation in conversations:
            question = conversation.question
            sql = question.sql
            predicted_sql = conversation.predicted_sql

            # Check if question is unanswerable (unsolvable)
            is_unanswerable = isinstance(question, QuestionUnanswerable) and not question.category.is_solvable()

            if predicted_sql is None:
                # System did not produce SQL
                if is_unanswerable:
                    # Expected: unanswerable questions should produce feedback, not SQL
                    conversation.solved = None
                else:
                    # Failed: answerable/solvable questions should produce SQL
                    conversation.solved = False
                continue

            if sql is None:
                # System produced SQL but there's no ground truth to compare against
                conversation.solved = False
                continue

            # Both predicted and ground truth SQL exist — compare them
            result = self.db.compare_query_results(
                db_id=question.db_id,
                predicted_sql=predicted_sql,
                ground_truth_sql=sql
            )

            conversation.solved = result or False  # Treat None as False
