# ABISS: Evaluating Text-to-SQL Systems Through Interactive Ambiguity Resolution

This repository contains the code and benchmarking framework for our VLDB 2026 paper on evaluating text-to-SQL systems with a comprehensive taxonomy of unanswerable and ambiguous questions.

## Overview

Modern text-to-SQL systems struggle with questions that are unanswerable, ambiguous, or require clarification. This work introduces:

1. **A comprehensive taxonomy** of 12 question categories that characterize different types of answerable and unanswerable questions
2. **An automated generation pipeline** for creating diverse, high-quality benchmark questions across multiple categories, styles, and difficulty levels
3. **An interactive benchmarking framework** for evaluating how text-to-SQL systems handle clarification dialogues with users

### Question Categories

Our taxonomy includes:

**Answerable Questions:**
- **Answerable**: Questions that can be directly answered from the database schema

**Unanswerable-Solvable Questions** (can be resolved through clarification):
- **Lexical Vagueness**: Terms used are vague or imprecise (e.g., "recent", "high", "popular")
- **Semantic Mapping - Entity Ambiguity**: Terms map to attributes from multiple different entities or tables
- **Semantic Mapping - Lexical Overlap**: Natural language terms overlap with multiple schema attributes within the same entity
- **Structure Ambiguity - Attachment**: Unclear which entity a modifier attaches to in conjunctions
- **Structure Ambiguity - Scope**: Unclear scope of quantifiers (e.g., "each", "every", "all")
- **Conflicting Knowledge**: Multiple contradictory definitions exist in the knowledge base
- **Missing User Knowledge**: User-specific context is required but not available

**Unanswerable-Unsolvable Questions** (cannot be resolved):
- **Improper Question**: Question is nonsensical, malformed, or unrelated to the database domain
- **Missing External Knowledge**: Question requires domain-specific facts, conventions, or policies not in the database
- **Missing Schema Entities**: Required entities or attributes are completely absent from the schema
- **Missing Schema Relationships**: Required entities exist but lack foreign keys or relationships to connect them

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (for VLLM models)
- Sufficient disk space for model weights

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd taxonomy

# Install dependencies
pip install -r requirements.txt

# Install vLLM (for model inference)
pip install vllm
```

## Quick Start

### 1. Question Generation

Generate a dataset of questions across different categories, styles, and difficulty levels:

```bash
python do_question_generation.py \
    --db_name bird_dev \
    --db_root_path ../datasets/bird_dev/dev_databases \
    --model_names ../models/Qwen2.5-32B-Instruct ../models/Mistral-Small-3.2-24B-Instruct-2506 ../models/gemma-3-27b-it \
    --n_samples 5 \
    --tensor_parallel_size 2 \
    --question_path ../datasets/bird_dev/dev.json \
    --intermediate_results_folder ./intermediate_results \
    --output_path results/question_generation/dev_generated_questions.json \
    --categories Answerable LexicalVagueness MissingSchemaEntities \
    --verbose
```

**Key Parameters:**
- `--db_root_path`: Root directory containing database files
- `--db_name`: Name of the database to use (e.g., spider, bird_dev)
- `--model_names`: List of LLM model paths for generation (space-separated)
- `--n_samples`: Number of questions to generate per category per model
- `--categories`: Specific categories to generate (omit to use all 12 categories)
  - Available: `Answerable`, `ConflictingKnowledge`, `ImproperQuestion`, `LexicalVagueness`, `MissingExternalKnowledge`, `MissingSchemaEntities`, `MissingSchemaRelationships`, `MissingUserKnowledge`, `SemanticMappingEntityAmbiguity`, `SemanticMappingLexicalOverlap`, `StructureAmbiguityAttachment`, `StructureAmbiguityScope`
- `--styles`: Question styles (formal, colloquial, imperative, interrogative, descriptive, concise)
- `--difficulties`: Question difficulties (simple, moderate, complex, highly_complex)
- `--limit_categories`: Only use specified categories for validation (not all categories)
- `--intermediate_results_folder`: Save intermediate results for debugging

### 2. Interactive Benchmarking

Evaluate a text-to-SQL system on generated questions with simulated user interactions:

```bash
python do_interaction.py \
    --db_name bird_dev \
    --db_root_path ../datasets/bird_dev/dev_databases \
    --model_names ../models/Qwen2.5-32B-Instruct ../models/Mistral-Small-3.2-24B-Instruct-2506 ../models/gemma-3-27b-it \
    --tensor_parallel_size 2 \
    --question_path results/question_generation/dev_generated_questions.json \
    --output_path results/interaction/dev_interactions.json \
    --max_steps 5 \
    --verbose
```

**Key Parameters:**
- `--question_path`: Path to the JSON file with generated questions
- `--model_names`: List of LLM model paths to evaluate (space-separated)
- `--max_steps`: Maximum number of interaction turns between system and user (default: 5)
- `--categories`: Specific categories to benchmark (should match generation categories)
- `--output_path`: Where to save benchmark results

The benchmark simulates user interactions with different:
- **User knowledge levels**: FULL (knows schema), NL (natural language only), NONE (minimal knowledge)
- **Category usage modes**: GROUND_TRUTH (give the true category to the text-to-sql system), PREDICTED (use system's classification), NO_CATEGORY (no category information given to the system)

## Project Structure

The codebase is organized into modular components:

- **`agents/`** - System agents for question classification and SQL generation
- **`benchmarks/`** - Interactive benchmarking framework
- **`categories/`** - Question category definitions (12 categories implementing the taxonomy)
- **`dataset_dataclasses/`** - Data structures for questions and benchmark results
- **`db_datasets/`** - Database interface supporting Spider, BIRD, and other benchmarks
- **`evaluators/`** - Evaluation metrics (recognition, classification, generation, feedback)
- **`generators/`** - Question generation pipeline with validation
- **`models/`** - LLM model interfaces (vLLM implementation)
- **`users/`** - Simulated user behavior for benchmarking
- **`validators/`** - Question validation checks
- **`utils/`** - Utility functions

**Entry Point Scripts:**
- `do_question_generation.py` - Generate questions
- `do_interaction.py` - Run interactive benchmarking
- `*.job` files - SLURM job scripts for HPC clusters

For detailed descriptions of each file and module, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Key Features

### Multi-Dimensional Question Generation

Questions are generated with fine-grained control over:
- **Category**: Type of answerability/ambiguity (12 categories in taxonomy)
- **Style**: Linguistic formulation
  - Formal: "Retrieve the names of students enrolled in the database course."
  - Colloquial: "Who's taking the database class?"
  - Imperative: "Show me the database students."
  - Interrogative: "Which students are enrolled in the database course?"
  - Descriptive: "The students enrolled in the database course."
  - Concise: "Database students."
- **Difficulty**: Complexity of reasoning required
  - Simple: Single table, basic conditions
  - Moderate: Multiple tables, simple joins
  - Complex: Multiple joins, aggregations, subqueries
  - Highly Complex: Nested subqueries, complex aggregations, multiple conditions
- **Database**: Target database schema (Spider, BIRD, etc.)

### Robust Validation Pipeline

Generated questions undergo comprehensive validation:
- ✅ **SQL executability**: For answerable questions, verify SQL is syntactically correct and executes successfully
- ✅ **Category correctness**: Validate that ambiguity/unanswerability matches the specified category
- ✅ **Duplicate detection**: Ensure questions are unique and non-redundant
- ✅ **Ground truth verification**: Confirm SQL produces expected results
- ✅ **Style and difficulty consistency**: Verify questions match their specified style and difficulty
- ✅ **Unsolvability check**: For unsolvable questions, confirm they cannot be resolved
- ✅ **Feedback quality**: Validate that clarification feedback is informative and actionable

### Interactive Evaluation Framework

The benchmark provides sophisticated interaction simulation:
- **Multi-turn dialogues**: Systems can ask clarification questions across multiple turns (configurable max_steps)
- **User knowledge level simulation**:
  - FULL: User understands database schema and can reference schema elements
  - NL: User uses natural language only, no schema knowledge
  - NONE: Minimal user knowledge, may not provide useful clarifications
- **Category usage modes**:
  - GROUND_TRUTH: System uses the true category from ground truth
  - PREDICTED: System must predict the category from the question
  - NO_CATEGORY: System operates without category information
- **Comprehensive metrics**: Recognition accuracy, classification accuracy, SQL correctness, interaction efficiency, feedback quality

### Efficient Batched Inference

The framework leverages vLLM for high-performance inference:
- Batched generation across multiple questions and categories
- Prefix caching to reduce redundant computation
- Tensor parallelism for multi-GPU scaling
- Configurable batch sizes to optimize memory usage

## Evaluation Metrics

The framework provides four key evaluators accessible through the `evaluators/` module:

1. **Recognition Evaluator** (`recognition.py`): 
   - Measures whether the system correctly identifies questions as answerable vs. unanswerable
   - Binary classification accuracy across all questions

2. **Classification Evaluator** (`classification.py`): 
   - Measures whether the system identifies the specific category of unanswerable questions
   - Multi-class classification accuracy (12 categories)
   - Confusion matrix analysis

3. **Generation Evaluator** (`generation.py`): 
   - Assesses SQL quality for answerable questions
   - Execution accuracy (does SQL run without errors?)
   - Semantic correctness (does SQL return correct results?)
   - Comparison with ground truth SQL

4. **Feedback Evaluator** (`feedback.py`): 
   - Evaluates the quality of clarification questions asked by the system
   - Relevance: Does the clarification question address the ambiguity?
   - Informativeness: Does it help resolve the question?
   - Specificity: Is it concrete and actionable?

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{taxonomy2026,
  title={},
  author={[Authors]},
  booktitle={},
  year={2026},
  volume={XX},
  number={XX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please contact giovanni.sullutrone@unimore.it or open an issue on GitHub.
