from agents.system import System
from dataset_dataclasses.benchmark import Conversation, SystemResponse
from dataset_dataclasses.question import Question
from categories.category import Category
from categories import get_category_by_name_and_subname
from models.model import Model
from db_datasets.db_dataset import DBDataset
from agents.prompts.system_category_prompt import get_category_classification_prompt, CategoryClassificationResponse, get_category_result
from agents.prompts.system_response_prompt import get_system_response_prompt, get_system_response_result


class SystemLLM(System):
    def __init__(self, agent_name: str, model: Model, db: DBDataset, categories: list[Category], max_steps: int) -> None:
        super().__init__(agent_name, categories, max_steps)
        self.model = model
        self.db = db

    def get_category(self, questions: list[Question]) -> list[Category]:
        prompts = [get_category_classification_prompt(self.db, question, self.categories) for question in questions]
        
        self.model.init()
        responses = self.model.generate_batch_with_constraints(
            prompts, 
            [CategoryClassificationResponse] * len(prompts)
        )
        self.model.close()
        
        categories = []
        for response in responses:
            name, subname = get_category_result(response)
            category = get_category_by_name_and_subname(name, subname, fuzzy=True)
            if category is None:
                raise ValueError(f"Invalid category returned: {name}, {subname}")
            categories.append(category)
        return categories
    
    def get_system_response(
        self, 
        conversations: list[Conversation], 
        categories_to_use: list[Category | None],
        current_steps: list[int]
    ) -> list[SystemResponse]:        
        # Generate prompts and get appropriate model classes for each conversation
        prompts_and_models = [
            get_system_response_prompt(self.db, conv, cat, current_step, self.max_steps, categories=self.categories)
            for conv, cat, current_step in zip(conversations, categories_to_use, current_steps)
        ]
        
        prompts = [p for p, _ in prompts_and_models]
        model_classes = [m for _, m in prompts_and_models]
        
        self.model.init()
        responses = self.model.generate_batch_with_constraints(
            prompts,
            model_classes
        )
        self.model.close()
        
        return [get_system_response_result(response) for response in responses]
