from typing import cast
from pydantic import BaseModel
from db_datasets.db_dataset import DBDataset
from validators.validator import Validator
from dataset_dataclasses.question import Question
from models.model import Model
from prompts.check_gt_prompt import get_gt_validation_prompt, CheckGTResponse, get_gt_validation_result


class CheckGT(Validator):
    def __init__(self, db: DBDataset, models: list[Model], max_tokens: int, max_gen_tokens: int) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.max_tokens: int = max_tokens
        self.max_gen_tokens: int = max_gen_tokens

    def validate(self, questions: list[Question]) -> list[bool]:
        prompts: list[str] = []
        
        for question in questions:
            prompt = get_gt_validation_prompt(self.db, question)
            prompts.append(prompt)
        
        valids: list[list[bool]] = [[] for _ in questions]
        valid_questions: list[bool] = [True] * len(questions)

        for model in self.models:
            model.init()

            lengths = model.get_token_lengths(prompts)
            valid_indices = []
            for i, length in enumerate(lengths):
                if length > self.max_tokens - (self.max_gen_tokens * 1.1):
                    valid_questions[i] = False
                else:
                    valid_indices.append(i)
            
            if valid_indices:
                valid_prompts = [prompts[i] for i in valid_indices]
                valid_constraints = [CheckGTResponse] * len(valid_indices)
                responses: list[BaseModel] = model.generate_batch_with_constraints(valid_prompts, cast(list[type[BaseModel]], valid_constraints))
                for j, i in enumerate(valid_indices):
                    is_valid = get_gt_validation_result(responses[j])
                    valids[i].append(is_valid)
            
            # For invalid prompts, append False vote
            for i in range(len(prompts)):
                if i not in valid_indices:
                    valids[i].append(False)
            model.close()

        # Majority voting across models
        final_valids: list[bool] = []
        for i, votes in enumerate(valids):
            if not valid_questions[i]:
                final_valids.append(False)
            else:
                yes_votes = sum(votes)
                no_votes = len(votes) - yes_votes
                final_valids.append(yes_votes > no_votes)
        
        return final_valids
