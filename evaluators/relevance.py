from evaluators.evaluator import Evaluator
from dataset_dataclasses.results import Conversation
from dataset_dataclasses.system import SystemResponseQuestion
from dataset_dataclasses.question import QuestionUnanswerable
from interactions.user import User


class Relevance(Evaluator):
    def __init__(self, user: User) -> None:
        self.user: User = user

    def evaluate(self, conversations: list[Conversation]) -> list[Conversation]:
        questions = [conversation.question for conversation in conversations]
        system_responses = [conversation.interactions[-1].system_response for conversation in conversations]

        # Get the indices of unanswerable questions with SystemResponseQuestion
        unanswerable_question_indices = [idx for idx, (question, system_response) in enumerate(zip(questions, system_responses)) 
                                         if isinstance(system_response, SystemResponseQuestion) and isinstance(question, QuestionUnanswerable)]
        if unanswerable_question_indices:
            conversations_to_judge: list[tuple[int, Conversation]] = [(idx, conversations[idx]) for idx in unanswerable_question_indices]
            judged_relevances = self.user.get_relevancy([conversation for _, conversation in conversations_to_judge])

            for idx, judged_conversation in zip(unanswerable_question_indices, judged_relevances):
                conversations[idx] = judged_conversation
        return conversations
