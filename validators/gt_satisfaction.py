import logging
from typing import cast
from pydantic import BaseModel
from db_datasets.db_dataset import DBDataset
from validators.validator import Validator
from dataset_dataclasses.question import Question
from models.model import Model
from validators.prompts.gt_satisfaction_prompt import get_gt_satisfaction_prompt, GTSatisfactionResponse, get_gt_satisfaction_result
from dataset_dataclasses.council_tracking import ValidationStageResult, QuestionVotes, ModelVote

logger = logging.getLogger(__name__)


class GTSatisfaction(Validator):
    def __init__(self, db: DBDataset, models: list[Model], max_tokens: int, max_gen_tokens: int) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.max_tokens: int = max_tokens
        self.max_gen_tokens: int = max_gen_tokens

    def validate(self, questions: list[Question]) -> ValidationStageResult:
        prompts: list[str] = []

        for question in questions:
            prompt = get_gt_satisfaction_prompt(self.db, question)
            prompts.append(prompt)

        valids: list[list[bool]] = [[] for _ in questions]
        valid_questions: list[bool] = [True] * len(questions)
        token_limit = self.max_tokens - (self.max_gen_tokens * 1.15)

        model_names: list[str] = []
        for model in self.models:
            model_names.append(model.model_name)
            model.init()

            lengths = model.get_token_lengths(prompts)
            valid_indices = []
            for i, length in enumerate(lengths):
                if length > token_limit:
                    valid_questions[i] = False
                    logger.warning(
                        "[GTSatisfaction] Skipping question %d due to token length: "
                        "%d tokens > limit %.0f (max_tokens=%d, max_gen_tokens=%d) | "
                        "model=%s | db_id=%s | category=%s | question='%.80s...'",
                        i, length, token_limit, self.max_tokens, self.max_gen_tokens,
                        model.model_name, questions[i].db_id,
                        questions[i].category.get_name(), questions[i].question,
                    )
                else:
                    valid_indices.append(i)
            
            if valid_indices:
                valid_prompts = [prompts[i] for i in valid_indices]
                valid_constraints = [GTSatisfactionResponse] * len(valid_indices)
                responses: list[BaseModel | None] = model.generate_batch_with_constraints_unsafe(valid_prompts, cast(list[type[BaseModel]], valid_constraints))
                for j, i in enumerate(valid_indices):
                    if responses[j] is None:
                        valids[i].append(False)
                        continue

                    is_valid = get_gt_satisfaction_result(cast(GTSatisfactionResponse, responses[j]))
                    valids[i].append(is_valid)
            
            # For invalid prompts, append False vote
            for i in range(len(prompts)):
                if i not in valid_indices:
                    valids[i].append(False)
            model.close()

        # Majority voting across models (ties resolve conservatively: question rejected)
        final_valids: list[bool] = []
        question_votes: list[QuestionVotes] = []
        for i, votes in enumerate(valids):
            if not valid_questions[i]:
                final_valids.append(False)
                question_votes.append(QuestionVotes(
                    question_index=i,
                    question_text=questions[i].question,
                    votes=[],
                    aggregate_result=False,
                    removed=True,
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
            stage_name="gt_satisfaction",
            validities=final_valids,
            question_votes=question_votes,
        )
