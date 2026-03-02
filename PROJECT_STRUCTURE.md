# Project Structure

This document describes the organization of the ABISS codebase. Each directory and file is documented below.

## Core Directories

### `agents/`
System agents that handle question classification and response generation during interactive benchmarking.
- **`system.py`**: Abstract base class defining the system agent interface
- **`system_llm.py`**: LLM-based system implementation with category classification, clarification generation, SQL generation, and feedback generation
- **`prompts/system_category_prompt.py`**: Prompts for question category classification
- **`prompts/system_response_prompt.py`**: Prompts for system response generation (SQL, clarification, or feedback)

### `benchmarks/`
Interactive benchmarking framework for evaluating text-to-SQL systems.
- **`benchmark.py`**: Main benchmark orchestrator that simulates multi-turn conversations between system agents and user agents, supporting different category usage modes (ground truth, predicted, no category) and balanced evaluation

### `categories/`
Question category definitions implementing the taxonomy (8 main categories, 13 subcategories).
- **`category.py`**: Abstract base class defining the category interface with methods for name, definition, examples, output schema, and question generation
- **`answerable.py`**: Answerable questions without external evidence
- **`answerable_with_evidence.py`**: Answerable questions requiring external evidence
- **`conflicting_knowledge.py`**: Questions with contradictory knowledge base definitions (ambiguous)
- **`improper_question.py`**: Nonsensical or off-topic questions (unanswerable)
- **`lexical_vagueness.py`**: Questions with vague or imprecise terms (ambiguous)
- **`missing_external_knowledge.py`**: Questions requiring domain facts not in the database (unanswerable)
- **`missing_schema_entities.py`**: Questions about entities absent from the schema (unanswerable)
- **`missing_schema_relationships.py`**: Questions requiring missing foreign keys or relationships (unanswerable)
- **`missing_user_knowledge.py`**: Questions requiring user-specific context (ambiguous)
- **`semantic_mapping_entity_ambiguity.py`**: Terms mapping to attributes in different entities (ambiguous)
- **`semantic_mapping_lexical_overlap.py`**: Terms overlapping with multiple attributes in the same entity (ambiguous)
- **`structure_ambiguity_attachment.py`**: Modifier attachment ambiguity in conjunctions (ambiguous)
- **`structure_ambiguity_scope.py`**: Quantifier scope ambiguity (ambiguous)
- **`__init__.py`**: Category registry with `get_all_categories()`, `get_category_by_name_and_subname()`, and `get_category_by_class_name()` lookup functions

### `dataset_dataclasses/`
Data structures for questions, benchmark results, and council tracking.
- **`question.py`**: `Question` and `QuestionUnanswerable` dataclasses, with enums for `QuestionStyle` (formal, colloquial, imperative, interrogative, descriptive, concise) and `QuestionDifficulty` (simple, moderate, complex, highly_complex)
- **`benchmark.py`**: Benchmark result structures: `Results`, `Conversation`, `Interaction`, `SystemResponse`, `CategoryUse`, and `RelevancyLabel`
- **`council_tracking.py`**: Model vote tracking for council consensus: `ModelVote`, `QuestionVotes`, `ValidationStageResult`, `GenerationTrackingReport`, `RelevancyVotes`, `TournamentVotes`, and `BenchmarkTrackingReport`

### `db_datasets/`
Database interface and schema management.
- **`db_dataset.py`**: Database interface supporting SQLite databases (BIRD, Spider, and custom datasets); handles schema loading, SQL execution with timeout, result comparison, and prompt generation
- **`sql_generation_prompts.py`**: Prompts and response models for SQL generation tasks
- **`sql_schema_prompts.py`**: Prompts for database schema understanding and context

### `evaluators/`
Evaluation metrics for system performance.
- **`evaluator.py`**: Abstract base evaluator interface
- **`recognition.py`**: Evaluates whether systems correctly identify question type (answerable, ambiguous, or unanswerable)
- **`classification.py`**: Evaluates whether systems identify the specific subcategory correctly
- **`generation.py`**: Assesses SQL correctness via execution accuracy (semantic equivalence of results)
- **`feedback.py`**: Evaluates feedback quality for unanswerable questions using council voting
- **`prompts/feedback_evaluation_prompt.py`**: Prompts for feedback quality assessment

### `generators/`
Question generation pipeline.
- **`generator.py`**: Main generator that creates questions using category-aware prompts with structured Pydantic output schemas
- **`chain.py`**: Generation chain that orchestrates multi-model generation and sequential validation
- **`prompts/generator_prompt.py`**: Prompts for category-aware question generation

### `models/`
LLM model interfaces.
- **`model.py`**: Abstract model interface
- **`model_vllm.py`**: vLLM implementation supporting batched inference, tensor parallelism, prefix caching, structured outputs via Pydantic, and configurable quantization (bitsandbytes, fp8, awq, gptq)

### `users/`
Simulated user behavior for interactive benchmarking.
- **`user.py`**: User agent that manages multi-turn interactions, coordinating relevancy assessment and response generation via council voting
- **`user_response.py`**: Generates user responses to system clarification questions with relevancy labels (relevant, technical, irrelevant)
- **`best_user_answer.py`**: Selects the best response from label-consistent candidates via pairwise tournament voting
- **`sql_preferences.py`**: Extracts secondary query preferences (ORDER BY, LIMIT, DISTINCT) from ground truth SQL in natural language
- **`prompts/user_response_prompt.py`**: Prompts for user response generation
- **`prompts/best_user_answer_prompt.py`**: Prompts for pairwise answer comparison in tournament selection

### `validators/`
Question validation pipeline (10 stages) ensuring generated question quality.
- **`validator.py`**: Abstract base validator interface
- **`duplicate_removal.py`**: Detects duplicate questions via exact match and value-masked SQL comparison
- **`sql_executability.py`**: Verifies SQL syntax, execution success, and non-empty results
- **`gt_satisfaction.py`**: Council voting to verify ground truth SQL correctly answers the question
- **`evidence_necessity.py`**: Verifies that evidence-bearing questions truly require their evidence (result set differs without it)
- **`ambiguity_verification.py`**: Council voting to confirm that ambiguous questions have multiple valid SQL interpretations
- **`unsolvability_verification.py`**: Council attempts to generate valid SQL for unanswerable questions; removes those that can be solved
- **`feedback_quality_check.py`**: Council voting on feedback accuracy, specificity, and actionability for unanswerable questions
- **`category_consistency.py`**: Council voting to confirm the question wins against all other categories of the same type
- **`difficulty_conformance.py`**: Rule-based SQL feature detection (joins, subqueries, window functions, CTEs) for difficulty validation
- **`style_conformance.py`**: Council voting to validate linguistic style matches the specification
- **`prompts/category_consistency_prompt.py`**: Prompts for pairwise category comparison
- **`prompts/feedback_quality_check_prompt.py`**: Prompts for feedback quality assessment
- **`prompts/gt_satisfaction_prompt.py`**: Prompts for ground truth verification
- **`prompts/style_conformance_prompt.py`**: Prompts for style classification

### `utils/`
Utility functions.
- **`dataclass_utils.py`**: Serialization and deserialization helpers for dataclasses and Pydantic models
- **`prompt_utils.py`**: Prompt formatting and conversation history construction
- **`style_and_difficulty_utils.py`**: Style and difficulty classification criteria definitions
- **`balancing.py`**: Balanced sampling utility for creating evaluation subsets with equal category or group representation

## Entry Point Scripts

- **`do_question_generation.py`**: Main script for generating questions with specified categories, styles, difficulties, and databases
- **`do_interaction.py`**: Main script for running interactive benchmarking with generated questions

## Analysis Scripts

- **`generate_result_charts.py`**: Generates per-category performance charts and CSV summaries from benchmark results
- **`generate_confusion_matrix.py`**: Generates classification confusion matrices from interaction results
- **`analyze_datasets.py`**: Analyzes generated datasets (category distributions, style distributions, semantic similarity)
- **`analyze_interactions.py`**: Analyzes interaction results (turn statistics, clarification rates, relevancy distributions)

## Configuration

- **`requirements.txt`**: Python dependencies
