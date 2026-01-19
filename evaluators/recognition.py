from categories.category import Category
from evaluators.evaluator import Evaluator
from dataset_dataclasses.benchmark import Conversation
from typing import cast


class Recognition(Evaluator):
    def evaluate(self, conversations: list[Conversation]) -> None:
        """
        Sets the recognition to True if the system correctly identified whether the question belongs to the same category group (i.e. same is_answerable and is_solvable).
        """
        question_categories = [conversation.question.category for conversation in conversations]
        predicted_categories = [conversation.predicted_category for conversation in conversations]

        assert all(pc is not None for pc in predicted_categories), "All conversations must have a predicted category for classification evaluation."
        predicted_categories = cast(list[Category], predicted_categories)

        results = [qc.is_answerable() == pc.is_answerable() and qc.is_solvable() == pc.is_solvable() for qc, pc in zip(question_categories, predicted_categories)]

        for conversation, result in zip(conversations, results):
            conversation.recognition = result