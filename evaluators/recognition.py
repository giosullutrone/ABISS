from evaluators.evaluator import Evaluator
from dataset_dataclasses.benchmark import Conversation, CategoryUse


class Recognition(Evaluator):
    def evaluate(self, conversations: list[Conversation]) -> None:
        """
        Sets the recognition to True if the system correctly identified whether the question belongs
        to the same category group (i.e. same is_answerable and is_solvable).
        Sets recognition to None when category_use is GROUND_TRUTH or NO_CATEGORY, or when
        no classification was performed (predicted_category is None).
        """
        for conversation in conversations:
            pc = conversation.predicted_category
            qc = conversation.question.category
            if pc is None or conversation.category_use in (CategoryUse.GROUND_TRUTH, CategoryUse.NO_CATEGORY):
                conversation.recognition = None
            else:
                conversation.recognition = qc.is_answerable() == pc.is_answerable() and qc.is_solvable() == pc.is_solvable()
