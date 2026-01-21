import argparse
import logging
from typing import Type
from dataset_dataclasses.question import Question, QuestionUnanswerable
from dataset_dataclasses.benchmark import UserKnowledgeLevel, CategoryUse
from db_datasets.db_dataset import DBDataset
from benchmarks.benchmark import Benchmark
from enum import Enum
from agents.system_llm import SystemLLM
from agents.system import System
import json
from models.model import Model
from models.model_vllm import ModelVLLM
from users.user import User
from categories import get_all_categories, get_category_by_class_name
import os


class Systems(Enum):
    DEFAULT = SystemLLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run interaction runner")
    parser.add_argument("--db_name", type=str, required=False, help="Name of the database to use")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to the database root")
    parser.add_argument("--question_path", type=str, required=True, help="Path to the questions file")
    parser.add_argument("--model_names", type=str, nargs='+', required=False, help="List of model names to use")
    parser.add_argument("--categories", type=str, nargs='+', required=False, help="List of category names to generate (if not specified, all categories will be used). Note: Use the same categories used to generate the dataset", default=None)
    parser.add_argument("--knowledge_levels", type=str, nargs='+', required=False, help="List of user knowledge levels to test (full, nl, none). If not specified, all levels will be used", default=None)
    parser.add_argument("--category_uses", type=str, nargs='+', required=False, help="List of category uses to test (ground_truth, predicted, no_category). If not specified, all uses will be used", default=None)
    parser.add_argument("--max_steps", type=int, required=False, help="Maximum number of interaction steps", default=5)
    parser.add_argument("--tensor_parallel_size", type=int, required=False, help="Tensor parallel size for VLLM models", default=1)
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the results", default="results.json")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    db_name: str = args.db_name
    db_root_path: str = args.db_root_path
    question_path: str = args.question_path
    model_names: list[str] = args.model_names
    category_names: list[str] | None = args.categories
    knowledge_level_names: list[str] | None = args.knowledge_levels
    category_use_names: list[str] | None = args.category_uses
    max_steps: int = args.max_steps
    tensor_parallel_size: int = args.tensor_parallel_size
    output_path: str = args.output_path

    # Normally we would use all categories
    categories = get_all_categories()

    # If the user specified categories, use only those
    if category_names is not None:
        categories = []
        for cat_name in category_names:
            # Try exact match first, then try with "Category" suffix
            category = get_category_by_class_name(cat_name)
            if category is None and not cat_name.endswith("Category"):
                category = get_category_by_class_name(f"{cat_name}Category")
            if category is None:
                raise ValueError(f"Category '{cat_name}' not found")
            categories.append(category)

    # Normally we would use all knowledge levels
    knowledge_levels = list(UserKnowledgeLevel)

    # If the user specified knowledge levels, use only those
    if knowledge_level_names is not None:
        knowledge_levels = []
        for level_name in knowledge_level_names:
            try:
                level = UserKnowledgeLevel(level_name.lower())
                knowledge_levels.append(level)
            except ValueError:
                raise ValueError(f"Knowledge level '{level_name}' not found. Valid levels: {[l.value for l in UserKnowledgeLevel]}")

    # Normally we would use all category uses
    category_uses = list(CategoryUse)

    # If the user specified category uses, use only those
    if category_use_names is not None:
        category_uses = []
        for use_name in category_use_names:
            try:
                use = CategoryUse(use_name.lower())
                category_uses.append(use)
            except ValueError:
                raise ValueError(f"Category use '{use_name}' not found. Valid uses: {[u.value for u in CategoryUse]}")

    ################################################################################
    ### System and Model Initialization
    system_class: Type[System] = SystemLLM
    model_system_name = "../models/Mistral-Small-3.2-24B-Instruct-2506"

    model_system = ModelVLLM(model_name=model_system_name,
                                sampling_kwargs={
                                    "max_tokens": 2048,
                                    "temperature": 0.15, # Suggested for Mistral-Small-3.2-24B-Instruct-2506
                                    "seed": 42,
                                },
                                model_kwargs={
                                    "max_model_len": 32000, 
                                    "max_num_batched_tokens": 32000,
                                    "enable_prefix_caching": True, 
                                    "enforce_eager": True,
                                    "tensor_parallel_size": tensor_parallel_size,
                                    "limit_mm_per_prompt": {"image": 0, "video": 0}, 
                                },
                                max_batch_with_text_size=100000)
    ################################################################################

    models_validator: list[Model] = [ModelVLLM(model_name=model,
                               sampling_kwargs={
                                   "max_tokens": 2048,
                                   "temperature": 0.0,
                                   "seed": 42,
                               },
                               model_kwargs={
                                   "max_model_len": 32000, 
                                   "max_num_batched_tokens": 32000,
                                   "enable_prefix_caching": True, 
                                   "enforce_eager": True,
                                   "tensor_parallel_size": tensor_parallel_size,
                                   "limit_mm_per_prompt": {"image": 0, "video": 0}, 
                               },
                               max_batch_with_text_size=100000) for model in model_names]

    db_dataset = DBDataset(db_root_path=db_root_path, db_name=db_name)

    system_instance = system_class("LLM", model=model_system, db=db_dataset, categories=categories, max_steps=max_steps)

    user_instance = User("test", 
                         db_dataset, 
                         models_validator, 
                         db_dataset.get_db_ids())

    runner = Benchmark(
        db_dataset=db_dataset,
        system=system_instance,
        user=user_instance,
        max_steps=max_steps,
        knowledge_levels=knowledge_levels,
        category_uses=category_uses
    )

    # Load raw question dicts and dispatch to the appropriate dataclass
    with open(question_path, 'r') as f:
        raw_questions = json.load(f)

    questions: list[Question] = []
    for d in raw_questions:
        try:
            question = QuestionUnanswerable.from_dict(d)
        except:
            question = Question(**d)
        questions.append(question)

    results = runner.run(questions=questions)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(json.dumps(results.to_dict(), indent=4))
