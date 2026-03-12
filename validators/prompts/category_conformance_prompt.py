from utils.prompt_utils import model_field_descriptions
from categories.category import Category
from dataset_dataclasses.question import Question, QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel, Field
from typing import Annotated, Literal


class CategoryConformanceResponse(BaseModel):
    belongs_to_category: Annotated[Literal["Yes", "No"], Field(description="'Yes' if the question genuinely belongs to the specified category, 'No' otherwise. Put only 'Yes' or 'No'.")]


def get_category_conformance_result(response: BaseModel) -> bool:
    """Returns True if the question belongs to the category, False otherwise."""
    validated = CategoryConformanceResponse.model_validate(response)
    return "yes" in validated.belongs_to_category.strip().lower()


def get_category_conformance_prompt(db: DBDataset, question: Question) -> str:
    category = question.category
    is_ambiguous = isinstance(question, QuestionUnanswerable) and question.category.is_solvable()

    prompt = "You are an expert in text-to-SQL ambiguity and unanswerability classification. " \
             "Your task is to determine whether a question genuinely belongs to a specific category " \
             "based on the category's definition, characteristics, and examples.\n\n"

    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=5) + "\n\n"

    prompt += "## Question Under Analysis\n"
    prompt += f"**Natural Language Question:** {question.question}\n"

    if question.evidence:
        prompt += f"**Evidence/Context:** {question.evidence}\n"

    if question.sql:
        prompt += f"**Ground Truth SQL:** {question.sql}\n"

        results = db.execute_query(
            db_id=question.db_id,
            sql_query=question.sql
        )
        if results is not None:
            prompt += f"**Query Results (first 5 rows):** {results[:5]}\n"

    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        if is_ambiguous:
            prompt += f"**Disambiguation Information** (clarifies the user's intended interpretation among the possible ones):\n{question.hidden_knowledge}\n"
        else:
            prompt += f"**Unsolvability Feedback** (explains why the question cannot be answered with the current schema/knowledge):\n{question.hidden_knowledge}\n"

    prompt += "\n## Category\n"
    prompt += f"**Name:** {category.get_name()}"
    if category.get_subname():
        prompt += f" - {category.get_subname()}"
    prompt += "\n"
    prompt += f"**Definition:** {category.get_definition()}\n"

    examples = category.get_examples()
    if examples:
        prompt += "**Examples:**\n"
        for example in examples:
            prompt += f"  - {example}\n"
    prompt += "\n"

    prompt += "## Classification Task\n"
    prompt += "Determine whether the question genuinely belongs to the specified category. " \
              "A question belongs to a category if it clearly exhibits the core characteristics " \
              "described in the category's definition.\n\n"

    prompt += "**Evaluation Criteria (in order of importance):**\n"
    prompt += "1. **Core Definition Alignment**: Does the question exhibit the fundamental characteristics described in the category's definition?\n"
    prompt += "2. **Distinctive Features**: Does the question display the specific, distinguishing features of this category?\n"
    prompt += "3. **Problem Type Match**: Does the underlying issue in the question match the type of problem this category describes?\n"

    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        if is_ambiguous:
            prompt += "4. **Disambiguation Alignment**: Does the disambiguation information match the nature of the ambiguity described by the category?\n"
        else:
            prompt += "4. **Feedback Alignment**: Does the unsolvability feedback match the specific limitation described by the category?\n"

    has_extra_criterion = isinstance(question, QuestionUnanswerable) and question.hidden_knowledge
    prompt += f"{'5' if has_extra_criterion else '4'}. **Example Similarity**: Is the question similar to the category's examples?\n\n"

    prompt += "**Answer 'Yes' if:** The question clearly and genuinely belongs to this category based on the criteria above.\n"
    prompt += "**Answer 'No' if:** The question does not fit this category, or fits it only superficially without exhibiting its core characteristics.\n\n"

    prompt += "## Response Format\n"
    prompt += "Provide a brief analysis addressing:\n"
    prompt += "1. **Question Analysis**: Identify the key characteristics of the question\n"
    prompt += "2. **Category Alignment**: How well does the question match the category's definition and examples?\n"
    prompt += "3. **Verdict**: Does it belong to this category?\n\n"
    prompt += "Keep your reasoning concise (approximately 100-200 words). Be decisive and objective.\n\n"
    prompt += "Then provide your final verdict as a JSON object with:\n"
    prompt += model_field_descriptions(CategoryConformanceResponse) + "\n"

    return prompt
