# Project Structure

The codebase is organized into modular components for generation, validation, benchmarking, and evaluation. This document provides detailed descriptions of each directory and file.

## Core Directories

### `agents/`
System agents that handle question classification and SQL generation.
- **`system.py`**: Abstract base class defining the system interface
- **`system_llm.py`**: LLM-based system implementation with category classification and clarification capabilities
- **`prompts/`**: System prompts for classification (`system_category_prompt.py`) and response generation (`system_response_prompt.py`)

### `benchmarks/`
Interactive benchmarking framework for evaluating text-to-SQL systems.
- **`benchmark.py`**: Main benchmark orchestrator that simulates multi-turn conversations between systems and users, testing different user knowledge levels and category usage modes

### `categories/`
Question category definitions implementing the taxonomy.
- **`category.py`**: Abstract base class defining category interface with methods for name, definition, examples, and question generation
- **`answerable.py`**: Answerable questions
- **`lexical_vagueness.py`**: Questions with vague terms (solvable)
- **`semantic_mapping_entity_ambiguity.py`**: Terms mapping to multiple entities (solvable)
- **`semantic_mapping_lexical_overlap.py`**: Terms with lexical overlap in schema (solvable)
- **`structure_ambiguity_attachment.py`**: Modifier attachment ambiguity (solvable)
- **`structure_ambiguity_scope.py`**: Quantifier scope ambiguity (solvable)
- **`conflicting_knowledge.py`**: Contradictory knowledge base definitions (solvable)
- **`missing_user_knowledge.py`**: User-specific context required (solvable)
- **`improper_question.py`**: Malformed or off-topic questions (unsolvable)
- **`missing_external_knowledge.py`**: Domain facts/conventions not in database (unsolvable)
- **`missing_schema_entities.py`**: Required tables/columns absent from schema (unsolvable)
- **`missing_schema_relationships.py`**: Required foreign keys/relationships missing (unsolvable)
- **`__init__.py`**: Exports `get_all_categories()`, `get_category_by_name_and_subname()`, and `get_category_by_class_name()` utility functions

### `dataset_dataclasses/`
Data structures for questions and benchmark results.
- **`question.py`**: Question dataclasses (`Question`, `QuestionUnanswerable`) with enums for `QuestionStyle` (formal, colloquial, imperative, interrogative, descriptive, concise) and `QuestionDifficulty` (simple, moderate, complex, highly_complex)
- **`benchmark.py`**: Benchmark result structures (`Results`, `Interaction`, `Conversation`, `SystemResponse`) and `CategoryUse`

### `db_datasets/`
Database interface and schema management.
- **`db_dataset.py`**: Database interface supporting Spider, BIRD, and other text-to-SQL benchmarks; handles schema loading, SQL execution, and evidence retrieval
- **`sql_generation_prompts.py`**: Prompts for SQL generation tasks
- **`sql_schema_prompts.py`**: Prompts for schema understanding

### `evaluators/`
Evaluation metrics for system performance.
- **`evaluator.py`**: Base evaluator interface
- **`recognition.py`**: Evaluates whether systems correctly identify answerable vs. unanswerable questions
- **`classification.py`**: Evaluates whether systems identify the specific category correctly
- **`generation.py`**: Assesses SQL quality (execution accuracy, semantic correctness)
- **`feedback.py`**: Evaluates clarification question quality and relevance
- **`prompts/`**: Evaluation-specific prompts

### `generators/`
Question generation pipeline with validation.
- **`generator.py`**: Main generator orchestrating question creation and validation across categories
- **`chain.py`**: Generation chain that sequences generation and validation steps
- **`generator_prompt.py`**: Prompts for question generation

### `models/`
LLM model interfaces.
- **`model.py`**: Abstract model interface
- **`model_vllm.py`**: vLLM implementation for efficient batched inference with support for tensor parallelism and prefix caching

### `users/`
Simulated user behavior for benchmarking.
- **`user.py`**: User simulator that responds to system clarification questions based on knowledge level
- **`user_answer.py`**: Generates user responses to clarification questions
- **`best_user_answer.py`**: Selects best response from multiple candidates
- **`question_relevancy.py`**: Evaluates relevancy of system questions
- **`schema_to_nl.py`**: Converts schema knowledge to natural language for user responses
- **`best_description.py`**: Generates natural descriptions of schema elements
- **`prompts/`**: User behavior prompts

### `validators/`
Question validation checks ensuring quality.
- **`validator.py`**: Base validator interface
- **`check_ambiguousness.py`**: Validates that generated ambiguity matches the category specification
- **`check_duplicate.py`**: Detects duplicate questions
- **`check_gt.py`**: Verifies ground truth correctness for answerable questions
- **`check_unsolvable.py`**: Validates that unsolvable questions are indeed unsolvable
- **`sql_executable.py`**: Verifies SQL validity and executability
- **`style_difficulty_check.py`**: Ensures questions match specified style and difficulty
- **`feedback_quality_check.py`**: Validates quality of clarification feedback
- **`category_comparison.py`**: Compares predicted vs. ground truth categories
- **`prompts/`**: Validation-specific prompts

### `utils/`
Utility functions.
- **`dataclass_utils.py`**: Serialization and deserialization helpers for dataclasses
- **`prompt_utils.py`**: Prompt formatting and construction utilities
- **`style_and_difficulty_utils.py`**: Utilities for question style and difficulty handling

## Entry Point Scripts

- **`do_question_generation.py`**: Script to generate questions with specified categories, styles, and difficulties
- **`do_interaction.py`**: Script to run interactive benchmarking with generated questions
- **`do_question_generation_array.job`**: SLURM job script for parallel generation across all categories
- **`do_interaction.job`**: SLURM job script for running benchmarks on HPC clusters
