from evaluators.evaluator import Evaluator
from dataset_dataclasses.benchmark import Conversation, CategoryUse


class Classification(Evaluator):
    def evaluate(self, conversations: list[Conversation]) -> None:
        """
        Sets classification to True if the category predicted matches the question's category.
        Sets classification to None when category_use is GROUND_TRUTH or NO_CATEGORY,
        or when no classification was performed (predicted_category is None).
        """
        for conversation in conversations:
            pc = conversation.predicted_category
            if pc is None or conversation.category_use in (CategoryUse.GROUND_TRUTH, CategoryUse.NO_CATEGORY):
                conversation.classification = None
            else:
                conversation.classification = conversation.question.category == pc
