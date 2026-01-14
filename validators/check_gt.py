from pydantic import BaseModel
from db_datasets.db_dataset import DBDataset
from validators.validator import Validator
from dataset_dataclasses.question import Question
from models.model import Model
from prompts.check_gt_prompt import get_gt_validation_prompt, CheckGTResponse, get_gt_validation_result


class CheckGT(Validator):
    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def validate(self, questions: list[Question]) -> list[bool]:
        prompts: list[str] = []
        
        for question in questions:
            prompt = get_gt_validation_prompt(self.db, question)
            prompts.append(prompt)
        
        valids: list[list[bool]] = [[] for _ in questions]

        for model in self.models:
            model.init()
            responses: list[BaseModel] = model.generate_batch_with_constraints(prompts, [CheckGTResponse] * len(prompts))
            model.close()

            for i, response in enumerate(responses):
                is_valid = get_gt_validation_result(response)
                valids[i].append(is_valid)

        # Majority voting across models
        final_valids: list[bool] = []
        for votes in valids:
            yes_votes = sum(votes)
            no_votes = len(votes) - yes_votes
            final_valids.append(yes_votes > no_votes)
        
        return final_valids
