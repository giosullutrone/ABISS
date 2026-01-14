from db_datasets.db_dataset import DBDataset
from validators.validator import Validator
from dataset_dataclasses.question import Question
from models.model import Model
from prompts.category_check_prompt import get_category_validation_prompt, CategoryCheckResponse, get_category_validation_result
from pydantic import BaseModel


class CategoryCheck(Validator):
    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def validate(self, questions: list[Question]) -> list[bool]:
        prompts: list[str] = []
        
        for question in questions:
            prompt = get_category_validation_prompt(self.db, question.category, question)
            prompts.append(prompt)
        
        valids: list[list[bool]] = [[] for _ in questions]

        for model in self.models:
            model.init()
            responses: list[BaseModel] = model.generate_batch_with_constraints(
                prompts, 
                [CategoryCheckResponse] * len(prompts)
            )
            model.close()

            for i, response in enumerate(responses):
                is_valid = get_category_validation_result(response)
                valids[i].append(is_valid)

        # Majority voting across models
        final_valids: list[bool] = []
        for votes in valids:
            yes_votes = sum(votes)
            no_votes = len(votes) - yes_votes
            final_valids.append(yes_votes > no_votes)        
        return final_valids
