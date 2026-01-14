from agents.system import System, SystemResponseQuestion, SystemResponseSQL
from interactions.user import User
from dataset_dataclasses.question import Question, QuestionUnanswerable
from dataset_dataclasses.results import Results, InteractionEvaluated, Conversation
from db_datasets.db_dataset import DBDataset

from evaluators.recognition import Recognition
from evaluators.classification import Classification
from evaluators.relevance import Relevance
from evaluators.generation import Generation


class Runner:
    def __init__(self, db_dataset: DBDataset, system: System, user: User, max_steps: int) -> None:
        self.db_dataset: DBDataset = db_dataset
        self.system: System = system
        self.user: User = user
        self.max_steps: int = max_steps

        self.evaluator_recognition = Recognition()
        self.evaluator_classification = Classification()
        self.evaluator_relevance = Relevance(user)
        self.evaluator_generation = Generation(db_dataset)

    def get_conversations_to_continue(self, conversations: list[tuple[int, Conversation]], step: int) -> list[tuple[int, Conversation]]:
        """
        Returns the list of conversations that need to continue to the next step.

        A conversation continues if none of the stopping conditions are met.
        Stopping conditions:
        - If we have reached the maximum number of steps
        - For step == 0:
            - If the system has generated a SQL query
            - If the question is answerable and the system has generated a question as response
        - For step > 0:
            - If the system has generated a SQL query
        """
        conversations_to_continue: list[tuple[int, Conversation]] = []
        for idx, conversation in conversations:
            last_interaction = conversation.interactions[-1]
            if step == 0:
                if isinstance(last_interaction.system_response, SystemResponseSQL):
                    continue
            
                if not isinstance(conversation.question, QuestionUnanswerable) and isinstance(last_interaction.system_response, SystemResponseQuestion):
                    continue
            else:
                if isinstance(last_interaction.system_response, SystemResponseSQL):
                    continue
            conversations_to_continue.append((idx, conversation))
        return conversations_to_continue

    def get_user_responses(self, conversations: list[Conversation]) -> list[Conversation]:
        """
        For each conversation, get the user response for the last interaction for unanswerable questions with question responses.
        """
        questions = [conversation.question for conversation in conversations]
        system_responses = [conversation.interactions[-1].system_response for conversation in conversations]

        # Only for unanswerable questions with question responses
        unanswerable_question_indices = [idx for idx, (question, system_response) in enumerate(zip(questions, system_responses)) 
                                         if isinstance(system_response, SystemResponseQuestion) and isinstance(question, QuestionUnanswerable)]
        if unanswerable_question_indices:
            conversations_updated = self.user.get_answers([conversations[idx] for idx in unanswerable_question_indices])
            for idx, updated_conversation in zip(unanswerable_question_indices, conversations_updated):
                conversations[idx] = updated_conversation
        return conversations

    def run_conversations_step(self, step: int, conversations: list[Conversation]) -> list[Conversation]:
        # System responses
        system_responses = self.system.get_system_responses(conversations)

        # Create partials conversations to evaluate
        partial_conversations: list[Conversation] = []
        for idx, conversation in enumerate(conversations):
            partial_conversation = Conversation(
                question=conversation.question,
                interactions=conversation.interactions + [InteractionEvaluated(system_response=system_responses[idx])]
            )
            partial_conversations.append(partial_conversation)

        # Analyze Recognition
        # If it is the first step, we need to evaluate recognition
        if step == 0:
            partial_conversations = self.evaluator_recognition.evaluate(partial_conversations)

        # Analyze Generation
        # If it is not the first step, we need to evaluate generation for SQL responses to see if the ambiguity is resolved
        if step > 0:
            partial_conversations = self.evaluator_generation.evaluate(partial_conversations)

        # Analyze Classification
        partial_conversations = self.evaluator_classification.evaluate(partial_conversations)

        # Analyze Relevance
        partial_conversations = self.evaluator_relevance.evaluate(partial_conversations)

        # Get User Responses
        partial_conversations = self.get_user_responses(partial_conversations)
        return partial_conversations

    def run(self, questions: list[Question]) -> Results:
        # Initialize conversations
        conversations: list[Conversation] = [Conversation(question=question, interactions=[]) for question in questions]
        # List of tuples (index, Conversation) of conversations that need to continue
        conversations_to_continue: list[tuple[int, Conversation]] = [(idx, conv) for idx, conv in enumerate(conversations)]

        for step in range(self.max_steps):
            conversations_current_step = [conversation for _, conversation in conversations_to_continue]
            if not conversations_current_step:
                break  # All conversations are complete

            conversations_updated = self.run_conversations_step(step, conversations_current_step)
            # Update the main conversations list with the updated conversations
            for (c2c_idx, (idx, _)), updated_conversation in zip(enumerate(conversations_to_continue), conversations_updated):
                conversations[idx] = updated_conversation
                conversations_to_continue[c2c_idx] = (idx, updated_conversation)

            # Update the conversations to continue for the next step
            conversations_to_continue = self.get_conversations_to_continue(conversations_to_continue, step)

        results = Results(
            agent_name=self.system.agent_name,
            user_models=[str(self.user)],
            dataset_name=self.db_dataset.db_name,
            conversations=conversations
        )
        return results