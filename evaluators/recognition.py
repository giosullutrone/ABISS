from evaluators.evaluator import Evaluator
from dataset_dataclasses.results import Conversation
from dataset_dataclasses.system import SystemResponseQuestion, SystemResponseSQL
from dataset_dataclasses.question import Question, QuestionUnanswerable


class Recognition(Evaluator):
    def evaluate(self, conversations: list[Conversation]) -> list[Conversation]:
        """
        Returns True if the system response correctly recognizes whether the question is answerable or unanswerable:
        - For answerable questions, the system response should be a SystemResponseSQL.
        - For unanswerable questions, the system response should be a SystemResponseQuestion.
        """
        questions = [conversation.question for conversation in conversations]
        system_responses = [conversation.interactions[-1].system_response for conversation in conversations]
        recognitions: list[bool | None] = [None] * len(conversations)

        for idx, (question, system_response) in enumerate(zip(questions, system_responses)):
            recognitions[idx] = (isinstance(system_response, SystemResponseSQL) and isinstance(question, Question) or 
                                 isinstance(system_response, SystemResponseQuestion) and isinstance(question, QuestionUnanswerable))
        
        for conversation, recognition in zip(conversations, recognitions):
            conversation.interactions[-1].recognition = recognition
        return conversations
