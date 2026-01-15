from abc import ABC, abstractmethod
from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty
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
        
        # Generate all prompts and constraints for all categories at once
        prompts: list[str] = []
        constraints: list[type[BaseModel]] = []
        metadata: list[tuple[Category, str, QuestionStyle, QuestionDifficulty]] = []  # Track (category, db_id, style, difficulty) for each prompt
        
        for category in categories:
            for db_id in db_ids:
                # Generate prompts for all combinations of style and difficulty
                for style in QuestionStyle:
                    for difficulty in QuestionDifficulty:
                        # Prepare the generation prompt
                        prompt = get_generation_prompt(
                            db=self.db,
                            is_solvable=self.solvable,
                            is_answerable=category.is_answerable(),
                            db_id=db_id,
                            name=category.get_name(),
                            definition=category.get_definition(),
                            examples=category.get_examples(),
                            output=category.get_output(),
                            question_style=style,
                            question_difficulty=difficulty
                        )
                        prompts.append(prompt)
                        constraints.append(category.get_output())
                        metadata.append((category, db_id, style, difficulty))
        
        # Use the model to generate all questions in a single call
        model.init()
        responses: list[BaseModel] = model.generate_batch_with_constraints(prompts, constraints)
        model.close()

        # Convert the responses into Question instances
        assert len(responses) == len(prompts) * self.n_samples, "Number of responses does not match number of prompts times n_samples."
        for idx, response in enumerate(responses):
            category, db_id, style, difficulty = metadata[idx // self.n_samples]
            question = category.get_question(db_id, response, style, difficulty)
            questions.extend(question)
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