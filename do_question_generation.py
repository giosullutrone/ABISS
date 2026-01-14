import argparse
import logging
from db_datasets.db_dataset import DBDataset
from generators.chain import Chain
from generators.generator_answerable import GeneratorAnswerable
from generators.generator_solvable import GeneratorSolvable
from generators.generator_unsolvable import GeneratorUnsolvable
from categories import *
import json
from models.model_vllm import ModelVLLM
from models.model import Model
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset question generation")
    parser.add_argument("--db_name", type=str, required=False, help="Name of the database to use")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to the database root")
    parser.add_argument("--question_path", type=str, required=True, help="Path to the questions file")
    parser.add_argument("--model_names", type=str, nargs='+', required=False, help="List of model names to use")
    parser.add_argument("--n_samples", type=int, required=False, help="Number of samples to generate per category per model", default=1)
    parser.add_argument("--tensor_parallel_size", type=int, required=False, help="Tensor parallel size for VLLM models", default=1)
    parser.add_argument("--intermediate_results_folder", type=str, required=False, help="Folder to save intermediate results", default=None)
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the results", default="dataset.json")
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
    n_samples: int = args.n_samples
    tensor_parallel_size: int = args.tensor_parallel_size
    intermediate_results_folder: str | None = args.intermediate_results_folder
    output_path: str = args.output_path

    db_dataset = DBDataset(db_root_path=db_root_path, db_name=db_name)

    models: list[Model] = [ModelVLLM(model_name=model,
                               sampling_kwargs={
                                   "n": n_samples,
                                   "max_tokens": 4096,
                                   "temperature": 0.7,
                                   "seed": 42,
                               },
                               model_kwargs={
                                   "max_model_len": 12000, 
                                   "max_num_batched_tokens": 12000,
                                   "enable_prefix_caching": True, 
                                   "enforce_eager": True,
                                   "tensor_parallel_size": tensor_parallel_size,
                                   "limit_mm_per_prompt": {"image": 0, "video": 0}, 
                               },
                               max_batch_with_text_size=100000) for model in model_names]

    models_validator: list[Model] = [ModelVLLM(model_name=model,
                               sampling_kwargs={
                                   "max_tokens": 4096,
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

    generator_answerable = GeneratorAnswerable(
        db=db_dataset,
        models=models,
        models_validator=models_validator,
        n_samples=n_samples,
        intermediate_results_folder=intermediate_results_folder
    )

    generator_solvable = GeneratorSolvable(
        db=db_dataset,
        models=models,
        models_validator=models_validator,
        n_samples=n_samples,
        intermediate_results_folder=intermediate_results_folder
    )
    generator_unsolvable = GeneratorUnsolvable(
        db=db_dataset,
        models=models,
        models_validator=models_validator,
        n_samples=n_samples,
        intermediate_results_folder=intermediate_results_folder
    )

    chain = Chain(
        models=models,
        generator_answerable=generator_answerable,
        generator_solvable=generator_solvable,
        generator_unsolvable=generator_unsolvable,
        categories=get_all_categories()
    )

    questions = chain.generate(
        db=db_dataset
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([q.to_dict() for q in questions], f, indent=4)
