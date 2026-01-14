import argparse
from typing import Type
from dataset_dataclasses.question import Question, QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from interactions.runner import Runner
from enum import Enum
from agents.system_llm import SystemLLM
from agents.system import System
import json
from models.model import Model
from models.model_vllm import ModelVLLM
from interactions.user import User
from prompts import UserKnowledgeLevel, UserAnswerStyle
from categories import get_all_categories
import os


class Systems(Enum):
    DEFAULT = SystemLLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run interaction runner")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to the database root")
    parser.add_argument("--question_path", type=str, required=True, help="Path to the questions file")
    parser.add_argument("--model_names", type=str, nargs='+', required=False, help="List of model names to use")
    parser.add_argument("--tensor_parallel_size", type=int, required=False, help="Tensor parallel size for VLLM models", default=1)
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the results", default="results.json")
    args = parser.parse_args()

    db_root_path: str = args.db_root_path
    question_path: str = args.question_path
    model_names: list[str] = args.model_names
    tensor_parallel_size: int = args.tensor_parallel_size
    output_path: str = args.output_path

    ### System and Model Initialization ###
    system_class: Type[System] = Systems.DEFAULT.value
    model_system_name = "../models/Mistral-Small-3.2-24B-Instruct-2506"

    model_system = ModelVLLM(model_name=model_system_name,
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
                                max_batch_with_text_size=100000) 

    models_validator: list[Model] = [ModelVLLM(model_name=model,
                               sampling_kwargs={
                                   "max_tokens": 2048,
                                   "temperature": 0.2,
                                   "seed": 42,
                                   "frequency_penalty": 0.0,
                                   "top_k": 1,
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

    db_dataset = DBDataset(db_root_path=db_root_path)
    system_instance = system_class(model=model_system, db_dataset=db_dataset, categories=get_all_categories())
    user_instance = User("test", 
                         db_dataset, 
                         models_validator, 
                         db_dataset.get_db_ids(), 
                         UserKnowledgeLevel.NL, 
                         UserAnswerStyle.PRECISE)

    runner = Runner(
        db_dataset=db_dataset,
        system=system_instance,
        user=user_instance,
        max_steps=5
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
