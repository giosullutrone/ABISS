from abc import ABC, abstractmethod
from dataset_dataclasses.question import Question
from categories.category import Category
from db_datasets.db_dataset import DBDataset
from models.model import Model
import json
import os
from prompts.generator_prompt import get_generation_prompt
from pydantic import BaseModel


class Generator(ABC):
    def __init__(self, db: DBDataset, models: list[Model], n_samples: int, solvable: bool, intermediate_results_folder: str | None) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.n_samples: int = n_samples
        self.solvable: bool = solvable
        self.intermediate_results_folder = intermediate_results_folder

    def save_intermediate_results(self, questions: list[Question], stage: str) -> None:
        if self.intermediate_results_folder is not None:
            os.makedirs(self.intermediate_results_folder, exist_ok=True)
            with open(os.path.join(self.intermediate_results_folder, f"{'solvable' if self.solvable else 'unsolvable'}_intermediate_{stage}.json"), "w", encoding="utf-8") as f:
                json.dump([q.to_dict() for q in questions], f, ensure_ascii=False, indent=4)

    def generate_for_model(self, model: Model, db_ids: list[str], categories: list[Category]) -> list[Question]:
        questions: list[Question] = []
        model.init()
        for category in categories:
            prompts: list[str] = []
            for db_id in db_ids:
                # Prepare the generation prompt
                prompt = get_generation_prompt(
                    db=self.db,
                    is_solvable=self.solvable,
                    db_id=db_id,
                    name=category.get_name(),
                    definition=category.get_definition(),
                    examples=category.get_examples(),
                    output=category.get_output()
                )
                prompts.append(prompt)

            # Use the model to generate the questions
            responses: list[BaseModel] = model.generate_batch_with_constraints(prompts, [category.get_output()] * len(prompts) * self.n_samples)

            # Convert the responses into Question instances
            # Take into account the db_id repetition due to n_samples
            assert len(responses) == len(prompts) * self.n_samples, "Number of responses does not match number of prompts times n_samples. Check the model sampling settings."
            for idx, response in enumerate(responses):
                question = category.get_question(db_ids[idx // self.n_samples], response)
                questions.extend(question)
        model.close()
        return questions

    def generate(self, db_ids: list[str], categories: list[Category]) -> list[Question]:
        all_questions: list[Question] = []
        for model in self.models:
            questions = self.generate_for_model(model, db_ids, categories)
            all_questions.extend(questions)
        return all_questions
    
    @abstractmethod
    def validate(self, questions: list[Question]) -> list[Question]:
        pass