import argparse
import logging
from categories.category import Category
from db_datasets.db_dataset import DBDataset
from generators.chain import Chain
from generators.generator import Generator
from categories import get_all_categories, get_category_by_class_name
from dataset_dataclasses.question import QuestionStyle, QuestionDifficulty, get_all_question_styles, get_all_question_difficulties
import json
from models.model_vllm import ModelVLLM
from models.model import Model
import os


def get_categories_styles_difficulties(
    category_names: list[str] | None,
    style_names: list[str] | None,
    difficulty_names: list[str] | None
) -> tuple[list[Category], list[QuestionStyle], list[QuestionDifficulty]]:
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

    # Normally we would use all styles
    styles = get_all_question_styles()

    # If the user specified styles, use only those
    if style_names is not None:
        styles = []
        for style_name in style_names:
            try:
                style = QuestionStyle(style_name.lower())
                styles.append(style)
            except ValueError:
                raise ValueError(f"Style '{style_name}' not found. Valid styles: {[s.value for s in QuestionStyle]}")

    # Normally we would use all difficulties
    difficulties = get_all_question_difficulties()

    # If the user specified difficulties, use only those
    if difficulty_names is not None:
        difficulties = []
        for diff_name in difficulty_names:
            try:
                difficulty = QuestionDifficulty(diff_name.lower())
                difficulties.append(difficulty)
            except ValueError:
                raise ValueError(f"Difficulty '{diff_name}' not found. Valid difficulties: {[d.value for d in QuestionDifficulty]}")
    return categories, styles, difficulties


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
    parser.add_argument("--categories", type=str, nargs='+', required=False, help="List of category names to generate (if not specified, all categories will be used)", default=None)
    parser.add_argument("--styles", type=str, nargs='+', required=False, help="List of question styles to generate (formal, colloquial, imperative, interrogative, descriptive, concise)", default=None)
    parser.add_argument("--difficulties", type=str, nargs='+', required=False, help="List of question difficulties to generate (simple, moderate, complex, highly_complex)", default=None)
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
    category_names: list[str] | None = args.categories
    style_names: list[str] | None = args.styles
    difficulty_names: list[str] | None = args.difficulties

    db_dataset = DBDataset(db_root_path=db_root_path, db_name=db_name)

    models: list[Model] = [ModelVLLM(model_name=model,
                               sampling_kwargs={
                                   "n": n_samples,
                                   "max_tokens": 2048,
                                   "temperature": 0.7,
                                   "seed": 42,
                                   "frequency_penalty": 0.2,
                               },
                               model_kwargs={
                                   "max_model_len": 18000, 
                                   "max_num_batched_tokens": 18000,
                                   "enable_prefix_caching": True, 
                                   "enforce_eager": True,
                                   "tensor_parallel_size": tensor_parallel_size,
                                   "limit_mm_per_prompt": {"image": 0, "video": 0}, 
                               },
                               max_batch_with_text_size=100000) for model in model_names]

    models_validator: list[Model] = [ModelVLLM(model_name=model,
                               sampling_kwargs={
                                   "max_tokens": 2048,
                                   "temperature": 0.0,
                                   "seed": 42,
                               },
                               model_kwargs={
                                   "max_model_len": 18000, 
                                   "max_num_batched_tokens": 18000,
                                   "enable_prefix_caching": True, 
                                   "enforce_eager": True,
                                   "tensor_parallel_size": tensor_parallel_size,
                                   "limit_mm_per_prompt": {"image": 0, "video": 0}, 
                               },
                               max_batch_with_text_size=100000) for model in model_names]

    generator = Generator(
        db=db_dataset,
        models=models,
        models_validator=models_validator,
        n_samples=n_samples,
        intermediate_results_folder=intermediate_results_folder
    )

    categories, styles, difficulties = get_categories_styles_difficulties(
        category_names,
        style_names,
        difficulty_names
    )

    chain = Chain(
        models=models,
        generator=generator,
        categories=categories,
        styles=styles,
        difficulties=difficulties
    )

    questions = chain.generate(
        db=db_dataset
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([q.to_dict() for q in questions], f, indent=4)
