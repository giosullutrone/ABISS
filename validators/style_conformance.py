from db_datasets.db_dataset import DBDataset
from validators.validator import Validator
from dataset_dataclasses.question import Question
from models.model import Model
from validators.prompts.style_conformance_prompt import (
    get_style_conformance_prompt,
    StyleConformanceResponse,
    get_style_conformance_result
)
from pydantic import BaseModel
from typing import cast
from dataset_dataclasses.council_tracking import ValidationStageResult, QuestionVotes, ModelVote


class StyleConformance(Validator):
    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def validate(self, questions: list[Question]) -> ValidationStageResult:
        prompts: list[str] = []
        
        for question in questions:
            prompt = get_style_conformance_prompt(self.db, question)
            prompts.append(prompt)
        
        valids: list[list[bool]] = [[] for _ in questions]
        model_names: list[str] = [m.model_name for m in self.models]

        for model in self.models:
            model.init()
            responses: list[BaseModel | None] = model.generate_batch_with_constraints_unsafe(prompts, cast(list[type[BaseModel]], [StyleConformanceResponse] * len(prompts)))
            model.close()

            for i, response in enumerate(responses):
                if response is None:
                    valids[i].append(False)
                    continue
                is_valid = get_style_conformance_result(response)
                valids[i].append(is_valid)

        # Majority voting across models (ties resolve conservatively: question rejected)
        final_valids: list[bool] = []
        question_votes: list[QuestionVotes] = []
        for i, votes in enumerate(valids):
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
            stage_name="style_conformance",
            validities=final_valids,
            question_votes=question_votes,
        )
