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
from typing import cast
from dataset_dataclasses.council_tracking import ValidationStageResult, QuestionVotes, ModelVote


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

    def validate(self, questions: list[Question]) -> ValidationStageResult:
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
        model_names: list[str] = [m.model_name for m in self.models]

        for model in self.models:
            model.init()
            
            responses: list[BaseModel | None] = model.generate_batch_with_constraints_unsafe(
                prompts, 
                cast(list[type[BaseModel]], [FeedbackQualityCheckResponse] * len(prompts))
            )
            model.close()
            
            # Map responses back to original indices
            for i, response in enumerate(responses):
                if response is None:
                    valids[i].append(False)
                    continue
                is_valid = get_feedback_quality_check_result(response)
                valids[i].append(is_valid)
        
        # Apply majority voting for questions that were checked
        # Questions that weren't checked (empty prompts) automatically pass
        final_valids: list[bool] = []
        question_votes: list[QuestionVotes] = []
        for i, votes in enumerate(valids):
            if len(votes) == 0:
                final_valids.append(True)
                question_votes.append(QuestionVotes(
                    question_index=i,
                    question_text=questions[i].question,
                    votes=[],
                    aggregate_result=True,
                    removed=False,
                ))
            else:
                yes_votes = sum(votes)
                no_votes = len(votes) - yes_votes
                passed = yes_votes > no_votes
                final_valids.append(passed)
                question_votes.append(QuestionVotes(
                    question_index=i,
                    question_text=questions[i].question,
                    votes=[
                        ModelVote(model_name=model_names[j], vote=votes[j])
                        for j in range(len(votes))
                    ],
                    aggregate_result=passed,
                    removed=not passed,
                ))

        return ValidationStageResult(
            stage_name="feedback_quality_check",
            validities=final_valids,
            question_votes=question_votes,
        )
