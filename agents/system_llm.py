from .system import System
from dataset_dataclasses.system import SystemResponseQuestion, SystemResponseSQL
from models.model import Model
from dataset_dataclasses.results import Conversation
from agents.system_prompts import get_interaction_prompt, SystemResponse, get_system_response_result
from db_datasets.db_dataset import DBDataset
from difflib import get_close_matches
from categories.category import Category
from categories import get_category_by_name, get_all_categories
from enum import Enum

categories_enum = Enum('CategoriesEnum', {cat.get_name(): cat for cat in get_all_categories()})


class SystemLLM(System):
    def __init__(self, db_dataset: DBDataset, model: Model, categories: list[Category]) -> None:
        super().__init__(agent_name=model.model_name)
        self.db_dataset: DBDataset = db_dataset
        self.model: Model = model
        self.categories: list[Category] = categories

    def get_response_from_response(self, response: str) -> SystemResponseQuestion | SystemResponseSQL:
        response_type, content, category_str = get_system_response_result(response)
        
        response_type = response_type.lower()

        if response_type == 'sql':
            return SystemResponseSQL(sql=content)
        elif response_type == 'question':
            # Convert category string to Category enum
            category = None
            if category_str:
                # Find the closest matching Category enum value by edit distance
                category_values = [cat.get_name() for cat in self.categories]
                closest_matches = get_close_matches(category_str, category_values, n=1, cutoff=0.6)
                if closest_matches:
                    category = get_category_by_name(closest_matches[0])
            return SystemResponseQuestion(question=content, category=category)
        else:
            raise ValueError(f"Invalid response_type: {response_type}. Must be 'SQL' or 'QUESTION'.")

    def get_system_responses(self, conversations: list[Conversation]) -> list[SystemResponseQuestion | SystemResponseSQL]:
        self.model.init()
        prompts = [get_interaction_prompt(self.db_dataset, conversation, self.categories) for conversation in conversations]
        responses = self.model.generate_batch_with_constraints(prompts, SystemResponse.model_json_schema())
        self.model.close()
        return [self.get_response_from_response(resp) for resp in responses]
