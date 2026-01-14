from evaluators.evaluator import Evaluator
from dataset_dataclasses.results import Conversation
from dataset_dataclasses.system import SystemResponseQuestion, SystemResponseSQL
from dataset_dataclasses.question import Question, QuestionUnanswerable


class Classification(Evaluator):
    def evaluate(self, conversations: list[Conversation]) -> list[Conversation]:
        """
        Returns True if the system response classification matches the question type:
        - For unanswerable questions, the system response should be a SystemResponseQuestion with the correct category.
        - For answerable questions, the system response should be a SystemResponseSQL.
        """
        questions = [conversation.question for conversation in conversations]
        system_responses = [conversation.interactions[-1].system_response for conversation in conversations]
        classifications: list[bool | None] = [None] * len(conversations)

        for idx, (question, system_response) in enumerate(zip(questions, system_responses)):
            if isinstance(system_response, SystemResponseQuestion) and isinstance(question, QuestionUnanswerable):
                classifications[idx] = system_response.category is not None and system_response.category == question.category
            elif isinstance(system_response, SystemResponseSQL) and isinstance(question, Question):
                classifications[idx] = True

        for conversation, classification in zip(conversations, classifications):
            conversation.interactions[-1].classification = classification
        return conversations
