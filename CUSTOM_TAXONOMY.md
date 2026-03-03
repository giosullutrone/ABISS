# Custom Taxonomy Guide

This guide explains how to create custom question categories and integrate them into the ABISS generation and interaction pipelines.

## Overview

All question categories in ABISS implement the abstract `Category` base class defined in `categories/category.py`. Each category describes a specific type of question (answerable, ambiguous, or unanswerable), defines its output schema as a Pydantic `BaseModel`, and specifies how generated outputs are converted into `Question` objects.

Categories are registered in `categories/__init__.py` and referenced by class name from the command line.

## Category Interface

Every category must implement the following static methods:

| Method | Return Type | Purpose |
|--------|-------------|---------|
| `get_name()` | `str` | Main category name (for instance, `"Lexical Vagueness"`) |
| `get_subname()` | `str \| None` | Optional subcategory name (for instance, `"Scope Ambiguity"`), or `None` |
| `get_definition()` | `str` | Detailed description of the category, used in generation prompts |
| `get_examples()` | `list[str] \| None` | Example questions illustrating this category |
| `is_answerable()` | `bool` | `True` if the question can be answered directly |
| `is_solvable()` | `bool` | `True` if the question can be resolved through clarification |
| `get_output()` | `type[BaseModel]` | Pydantic model defining the structured output schema for generation |
| `get_question(db_id, output, question_style, question_difficulty)` | `list[Question]` | Converts a generated output into one or more `Question` objects |

The combination of `is_answerable()` and `is_solvable()` determines how the category is treated:

| `is_answerable()` | `is_solvable()` | Question Type |
|---|---|---|
| `True` | `True` | Answerable: the question has a direct SQL answer |
| `False` | `True` | Ambiguous: the question has multiple valid interpretations, resolvable through clarification |
| `False` | `False` | Unanswerable: the question cannot be resolved even with clarification |

## Creating a Custom Category

Below is a complete example of a custom category for **Temporal Ambiguity**, where questions contain time references that could refer to different time frames.

### Step 1: Create the Category File

Create `categories/temporal_ambiguity.py`:

```python
from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class TemporalAmbiguityCategory(Category):
    class TemporalAmbiguityOutput(BaseModel):
        question: Annotated[str, Field(
            description="A natural language question containing a temporal reference "
            "that could refer to different time frames, such as 'last year' "
            "(calendar year vs. rolling 12 months) or 'this semester' "
            "(which semester depends on context)."
        )]
        hidden_knowledge_first_interpretation: Annotated[str, Field(
            description="A statement clarifying the temporal reference under "
            "the first interpretation. For instance: 'Last year refers to "
            "the 2024 calendar year (January to December).'"
        )]
        hidden_knowledge_second_interpretation: Annotated[str, Field(
            description="A statement clarifying the temporal reference under "
            "an alternative interpretation. For instance: 'Last year refers "
            "to the rolling 12-month period ending today.'"
        )]
        sql_first_interpretation: Annotated[str, Field(
            description="A valid SQL query answering the question under the "
            "first temporal interpretation."
        )]
        sql_second_interpretation: Annotated[str, Field(
            description="A valid SQL query answering the question under the "
            "second temporal interpretation."
        )]

    @staticmethod
    def get_name() -> str:
        return "Temporal Ambiguity"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return (
            "Temporal Ambiguity arises when a question contains a time reference "
            "that admits multiple valid interpretations depending on how the time "
            "frame is anchored. Common sources include relative expressions like "
            "'last year' (calendar year vs. rolling period), 'this quarter' "
            "(fiscal vs. calendar quarter), or 'recently' (which overlaps with "
            "Lexical Vagueness when the term is gradable, but falls here when "
            "the ambiguity is specifically about which time window is intended)."
        )

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "Show all orders from last year. (Temporal ambiguity: 'last year' "
            "could mean the previous calendar year or the past 12 months)",
            "List students who enrolled this semester. (Temporal ambiguity: "
            "'this semester' could refer to fall or spring depending on context)",
        ]

    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return TemporalAmbiguityCategory.TemporalAmbiguityOutput

    @staticmethod
    def get_question(
        db_id: str,
        output: BaseModel,
        question_style: "QuestionStyle",
        question_difficulty: "QuestionDifficulty",
    ) -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable

        assert isinstance(output, TemporalAmbiguityCategory.TemporalAmbiguityOutput)
        return [
            QuestionUnanswerable(
                db_id=db_id,
                category=TemporalAmbiguityCategory(),
                question=output.question,
                evidence=None,
                sql=sql,
                hidden_knowledge=hk,
                is_solvable=TemporalAmbiguityCategory.is_solvable(),
                question_style=question_style,
                question_difficulty=question_difficulty,
            )
            for sql, hk in [
                (output.sql_first_interpretation, output.hidden_knowledge_first_interpretation),
                (output.sql_second_interpretation, output.hidden_knowledge_second_interpretation),
            ]
        ]
```

Note that ambiguous categories (where `is_solvable()` returns `True`) typically return **multiple `Question` objects** from `get_question()`, one per valid interpretation. Each interpretation has its own SQL and hidden knowledge. Unanswerable categories instead return a single `Question` with feedback explaining why the question cannot be answered:

```python
# Unanswerable category pattern (single Question, no SQL)
@staticmethod
def get_question(db_id, output, question_style, question_difficulty):
    from dataset_dataclasses.question import QuestionUnanswerable
    return [QuestionUnanswerable(
        db_id=db_id,
        category=MyUnanswerableCategory(),
        question=output.question,
        evidence=None,
        sql=None,
        hidden_knowledge=output.feedback,
        is_solvable=False,
        question_style=question_style,
        question_difficulty=question_difficulty,
    )]
```

### Step 2: Register the Category

Edit `categories/__init__.py` to add three things:

1. **Import** the new category:
```python
from categories.temporal_ambiguity import TemporalAmbiguityCategory
```

2. **Add to the registry** in `get_all_categories()`:
```python
def get_all_categories() -> list[Category]:
    return [
        # ... existing categories ...
        TemporalAmbiguityCategory(),
    ]
```

3. **Add to exports** in `__all__`:
```python
__all__ = [
    # ... existing exports ...
    "TemporalAmbiguityCategory",
]
```

### Step 3: Use in the Generation Pipeline

Generate questions using only your new category:

```bash
python do_question_generation.py \
    --db_root_path datasets/bird_dev/dev_databases \
    --model_names models/Qwen2.5-32B-Instruct \
    --categories TemporalAmbiguityCategory \
    --limit_categories \
    --n_samples 5 \
    --output_path results/temporal_ambiguity_questions.json \
    --verbose
```

The `--limit_categories` flag restricts validation to only the specified categories (instead of comparing against all categories during consistency checks).

To generate your custom category alongside existing ones:

```bash
python do_question_generation.py \
    --db_root_path datasets/bird_dev/dev_databases \
    --model_names models/Qwen2.5-32B-Instruct \
    --categories TemporalAmbiguityCategory LexicalVaguenessCategory \
    --n_samples 5 \
    --output_path results/mixed_questions.json \
    --verbose
```

### Step 4: Use in the Interaction Pipeline

Run interactive benchmarking on the generated questions:

```bash
python do_interaction.py \
    --db_name bird_dev \
    --db_root_path datasets/bird_dev/dev_databases \
    --question_path results/temporal_ambiguity_questions.json \
    --model_names models/Qwen2.5-32B-Instruct \
    --output_path results/temporal_ambiguity_interactions.json \
    --verbose
```

## How Validators Interact with Categories

The `is_answerable()` and `is_solvable()` flags on your category control which validation stages are applied to generated questions. The stages run in the following order:

| Stage | Validation | Applies When |
|-------|-----------|--------------|
| 1 | Duplicate Removal | All categories |
| 2 | SQL Executability | Answerable and ambiguous categories (`is_answerable()=True` or `is_solvable()=True`) |
| 3 | Ground Truth Satisfaction | Answerable and ambiguous categories (`is_answerable()=True` or `is_solvable()=True`) |
| 4 | Evidence Necessity | `AnswerableWithEvidenceCategory` only (hardcoded `isinstance` check) |
| 5 | Ambiguity Verification | Ambiguous categories (`is_answerable()=False`, `is_solvable()=True`) |
| 6 | Unsolvability Verification | Unanswerable categories (`is_answerable()=False`, `is_solvable()=False`) |
| 7 | Feedback Quality Check | Unanswerable categories (`is_answerable()=False`, `is_solvable()=False`) |
| 8 | Category Consistency | Ambiguous and unanswerable categories (excludes answerable) |
| 9 | Difficulty Conformance | All categories |
| 10 | Style Conformance | All categories |

When designing a custom category, ensure that `is_answerable()` and `is_solvable()` accurately reflect the nature of your category, as these flags determine the full validation path.
