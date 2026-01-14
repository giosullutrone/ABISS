from evaluators.evaluator import Evaluator
from dataset_dataclasses.results import Conversation
from dataset_dataclasses.system import SystemResponseQuestion, SystemResponseSQL
from dataset_dataclasses.question import QuestionUnanswerable


class Recognition(Evaluator):
    def evaluate(self, conversations: list[Conversation]) -> list[Conversation]:
        questions = [conversation.question for conversation in conversations]
        system_responses = [conversation.interactions[-1].system_response for conversation in conversations]
        recognitions: list[bool | None] = [None] * len(conversations)

        for idx, (question, system_response) in enumerate(zip(questions, system_responses)):
            recognitions[idx] = (isinstance(system_response, SystemResponseSQL) and not isinstance(question, QuestionUnanswerable) or 
                                 isinstance(system_response, SystemResponseQuestion) and isinstance(question, QuestionUnanswerable))
        
        for conversation, recognition in zip(conversations, recognitions):
            conversation.interactions[-1].recognition = recognition
        return conversations
