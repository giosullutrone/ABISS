from db_datasets.db_dataset import DBDataset
from validators.validator import Validator
from dataset_dataclasses.question import Question
from models.model import Model
from pydantic import BaseModel
from validators.prompts.feedback_quality_check_prompt import (
    get_feedback_quality_check_prompt,
    FeedbackQualityCheckResponse,
    get_feedback_quality_check_result
)


class FeedbackQualityCheck(Validator):
    """
    Validator that checks if the feedback for unsolvable questions correctly
    explains why they cannot be answered.
    
    Uses majority voting across multiple models to determine if the feedback
    is valid, clear, and correctly identifies the type of unsolvability.
    """
    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def validate(self, questions: list[Question]) -> list[bool]:
        """
        Validate that feedback for unsolvable questions is correct and clear.
        
        Returns a list of booleans indicating which questions have valid feedback.
        Only evaluates unsolvable questions with feedback in hidden_knowledge.
        """
        prompts: list[str] = []

        for question in questions:
            prompt = get_feedback_quality_check_prompt(self.db, question)
            prompts.append(prompt)
        
        # Collect votes from each model (only for questions that need validation)
        valids: list[list[bool]] = [[] for _ in questions]
        
        for model in self.models:
            model.init()
            
            responses: list[BaseModel] = model.generate_batch_with_constraints(
                prompts, 
                [FeedbackQualityCheckResponse] * len(prompts)
            )
            model.close()
            
            # Map responses back to original indices
            for i, response in enumerate(responses):
                is_valid = get_feedback_quality_check_result(response)
                valids[i].append(is_valid)
        
        # Apply majority voting for questions that were checked
        # Questions that weren't checked (empty prompts) automatically pass
        final_valids: list[bool] = []
        for i, votes in enumerate(valids):
            if len(votes) == 0:
                # No validation needed (answerable or solvable question, or no feedback)
                final_valids.append(True)
            else:
                # Majority voting
                yes_votes = sum(votes)
                no_votes = len(votes) - yes_votes
                final_valids.append(yes_votes > no_votes)
        
        return final_valids
