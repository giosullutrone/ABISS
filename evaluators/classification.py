from evaluators.evaluator import Evaluator
from dataset_dataclasses.benchmark import Conversation


class Classification(Evaluator):
    def evaluate(self, conversations: list[Conversation]) -> None:
        """
        Sets classification to True if the category predicted matches the question's category
        """
        question_categories = [conversation.question.category for conversation in conversations]
        predicted_categories = [conversation.predicted_category for conversation in conversations]

        assert not any(pc is None for pc in predicted_categories), "All conversations must have a predicted category for classification evaluation."

        results = [qc == pc for qc, pc in zip(question_categories, predicted_categories)]

        for conversation, result in zip(conversations, results):
            conversation.classification = result