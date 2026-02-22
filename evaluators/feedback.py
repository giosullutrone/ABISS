from evaluators.evaluator import Evaluator
from dataset_dataclasses.benchmark import Conversation
from dataset_dataclasses.question import QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from models.model import Model
from evaluators.prompts.feedback_evaluation_prompt import (
    get_feedback_evaluation_prompt,
    FeedbackEvaluationResponse,
    get_feedback_evaluation_result
)
from pydantic import BaseModel


class Feedback(Evaluator):
    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def evaluate(self, conversations: list[Conversation]) -> None:
        """
        Set the explained flag to True if the system's feedback matches the expected feedback
        for unsolvable questions. Uses majority voting across multiple models.
        
        Sets explained to False if:
        - The question is not unsolvable
        - The system did not provide feedback
        - The majority of models say the feedback doesn't match
        """
        # Identify which conversations are evaluable (unsolvable with system feedback)
        evaluable_indices: list[int] = []
        for i, conversation in enumerate(conversations):
            question = conversation.question

            # Check if question is unsolvable
            is_unanswerable = isinstance(question, QuestionUnanswerable) and not question.category.is_solvable()

            if not is_unanswerable:
                # Non-unanswerable questions: explained is N/A when no feedback was generated
                if conversation.predicted_feedback is None:
                    conversation.explained = None
                else:
                    conversation.explained = False
                continue

            # At this point, question is guaranteed to be QuestionUnanswerable
            assert isinstance(question, QuestionUnanswerable)

            # Check if system provided feedback
            if conversation.predicted_feedback is None:
                conversation.explained = False
                continue

            # This conversation can be evaluated
            evaluable_indices.append(i)
        
        # If no conversations are evaluable, we're done
        if not evaluable_indices:
            return
        
        # Generate prompts for evaluable conversations
        prompts: list[str] = []
        for idx in evaluable_indices:
            conversation = conversations[idx]
            prompt = get_feedback_evaluation_prompt(self.db, conversation)
            prompts.append(prompt)
        
        # Collect votes from each model
        votes: list[list[bool]] = [[] for _ in evaluable_indices]
        
        for model in self.models:
            model.init()
            responses: list[BaseModel] = model.generate_batch_with_constraints(
                prompts, 
                [FeedbackEvaluationResponse] * len(prompts)
            )
            model.close()
            
            for i, response in enumerate(responses):
                matches = get_feedback_evaluation_result(response)
                votes[i].append(matches)
        
        # Apply majority voting (ties resolve conservatively: feedback not accepted)
        for i, idx in enumerate(evaluable_indices):
            yes_votes = sum(votes[i])
            no_votes = len(votes[i]) - yes_votes
            conversations[idx].explained = yes_votes > no_votes
