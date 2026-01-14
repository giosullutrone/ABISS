from db_datasets.db_dataset import DBDataset
from validators.validator import Validator
from dataset_dataclasses.question import Question
from models.model import Model
from categories.category import Category
from prompts.category_check_prompt import get_category_validation_prompt, CategoryCheckResponse, get_category_validation_result
from pydantic import BaseModel


class CheckOtherCategories(Validator):
    def __init__(self, db: DBDataset, models: list[Model], categories: list[Category]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.categories: list[Category] = categories

    def validate(self, questions: list[Question]) -> list[bool]:
        # For each question, check if it fits any OTHER category (not its own)
        # If majority of models say it fits another category, mark as invalid
        
        # Build prompts for all (question, other_category) pairs once
        prompts: list[str] = []
        prompt_to_question_idx: list[int] = []
        
        for q_idx, question in enumerate(questions):
            for category in self.categories:
                # Skip the question's own category
                if category == question.category:
                    continue
                
                prompt = get_category_validation_prompt(self.db, category, question)
                prompts.append(prompt)
                prompt_to_question_idx.append(q_idx)
        
        valids: list[list[bool]] = [[] for _ in questions]

        for model in self.models:
            # Batch generate for all prompts
            model.init()
            responses: list[BaseModel] = model.generate_batch_with_constraints(prompts, [CategoryCheckResponse] * len(prompts))
            model.close()

            # Process responses: group by question and check if ANY other category fits
            question_other_category_results: list[list[bool]] = [[] for _ in questions]
            for i, response in enumerate(responses):
                is_valid = get_category_validation_result(response)
                q_idx = prompt_to_question_idx[i]
                question_other_category_results[q_idx].append(is_valid)
            
            # For each question, if ANY other category got "Yes", mark as invalid for this model
            for q_idx, other_category_results in enumerate(question_other_category_results):
                fits_other_category = any(other_category_results)
                valids[q_idx].append(not fits_other_category)  # Valid if it does NOT fit other categories

        # Majority voting across models
        final_valids: list[bool] = []
        for votes in valids:
            yes_votes = sum(votes)
            no_votes = len(votes) - yes_votes
            final_valids.append(yes_votes > no_votes)
        
        return final_valids
