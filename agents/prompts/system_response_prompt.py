from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel, Field, model_validator
from typing import Annotated
from utils.prompt_utils import model_field_descriptions, get_conversation_history_prompt
from dataset_dataclasses.benchmark import Conversation, SystemResponse
from categories.category import Category


class SystemResponseModel(BaseModel):
    system_question: Annotated[str | None, Field(description="A clear, concise clarification question to ask the user that will help resolve ambiguity or obtain missing information needed to generate a SQL query. Should be None if you're providing SQL or feedback instead.")]
    system_sql: Annotated[str | None, Field(description="A valid SQLite query that correctly answers the user's question based on the database schema. Should be None if you're asking a question or providing feedback instead.")]
    system_feedback: Annotated[str | None, Field(description="A clear, helpful explanation of why the question cannot be answered with the current database schema and what would be needed to answer it. Should be None if you're asking a question or providing SQL instead.")]

    @model_validator(mode='after')
    def exactly_one_field(self) -> 'SystemResponseModel':
        fields = [self.system_question, self.system_sql, self.system_feedback]
        non_null = sum(1 for f in fields if f is not None)
        if non_null != 1:
            raise ValueError(f"Exactly one field must be non-null, got {non_null}")
        return self

class SystemResponseModelLimited(BaseModel):
    system_sql: Annotated[str | None, Field(description="A valid SQLite query that correctly answers the user's question based on the database schema. Should be None if you're providing feedback instead.")]
    system_feedback: Annotated[str | None, Field(description="A clear, helpful explanation of why the question cannot be answered with the current database schema and what would be needed to answer it. Should be None if you're providing SQL instead.")]

    @model_validator(mode='after')
    def exactly_one_field(self) -> 'SystemResponseModelLimited':
        fields = [self.system_sql, self.system_feedback]
        non_null = sum(1 for f in fields if f is not None)
        if non_null != 1:
            raise ValueError(f"Exactly one field must be non-null, got {non_null}")
        return self


def get_system_response_result(response: BaseModel) -> SystemResponse:
    """Convert the model response to a SystemResponse object."""
    # Handle both full and limited response models
    if hasattr(response, 'system_question'):
        validated = SystemResponseModel.model_validate(response)
        return SystemResponse(
            system_question=validated.system_question,
            system_sql=validated.system_sql,
            system_feedback=validated.system_feedback
        )
    else:
        validated = SystemResponseModelLimited.model_validate(response)
        return SystemResponse(
            system_question=None,
            system_sql=validated.system_sql,
            system_feedback=validated.system_feedback
        )


def _get_category_type_label(category: Category) -> str:
    """Return a human-readable type label for a category."""
    if category.is_answerable():
        return "Answerable"
    elif category.is_solvable():
        return "Ambiguous"
    else:
        return "Unanswerable"


def get_system_response_prompt(
    db: DBDataset,
    conversation: Conversation,
    predicted_category: Category | None,
    current_step: int,
    max_steps: int,
    categories: list[Category] | None = None
) -> tuple[str, type[BaseModel]]:
    """
    Generate a prompt for the system to produce a response.

    The system must choose exactly ONE of: system_question, system_sql, or system_feedback.

    Args:
        db: Database dataset
        conversation: Current conversation
        predicted_category: Predicted category (can be None)
        current_step: Current step number (0-indexed, 0 = first interaction)
        max_steps: Maximum number of clarification questions allowed before final response required
        categories: Full list of taxonomy categories (used in NO_CATEGORY mode)

    Returns:
        A tuple of (prompt, model_class) where model_class is the appropriate Pydantic model
        to use for constrained generation.
    """
    question = conversation.question
    has_history = len(conversation.interactions) > 0
    is_answerable = predicted_category.is_answerable() if predicted_category else None
    is_solvable = predicted_category.is_solvable() if predicted_category else None
    is_final_step = current_step >= max_steps

    # ── 1. Role ──────────────────────────────────────────────────────────
    prompt = (
        "You are an expert text-to-SQL assistant. Your job is to help a user get the correct "
        "answer from a database by generating SQL, asking clarification questions when the "
        "question is ambiguous, or explaining why a question cannot be answered.\n\n"
    )

    # ── 2. Question Types Guide ──────────────────────────────────────────
    prompt += (
        "## Question Types and Expected Behavior\n"
        "Every user question falls into one of three types. Your response must match the type:\n\n"
        "1. **Answerable** — The question can be directly translated into SQL with the given schema and context. "
        "You should generate a SQL query without asking clarification questions.\n"
        "2. **Ambiguous** — The question has a genuine natural-language ambiguity (e.g., a vague term, "
        "an unclear entity reference, a structural reading that could go two ways). "
        "You should ask a focused clarification question that targets the specific ambiguity, "
        "then generate SQL once the user clarifies. Do NOT ask about information already available "
        "in the schema or evidence.\n"
        "3. **Unanswerable** — The question requires data, relationships, or external knowledge that "
        "the database does not contain. No amount of clarification can fix this. "
        "You should provide feedback explaining exactly what is missing and why the question "
        "cannot be answered.\n\n"
    )

    # ── 3. Question and Context ──────────────────────────────────────────
    prompt += "## User Question\n"
    prompt += f"**Question:** {question.question}\n"
    if question.evidence:
        prompt += f"**Evidence:** {question.evidence}\n"
    if has_history:
        prompt += "\n### Conversation History\n"
        prompt += get_conversation_history_prompt(conversation)
    prompt += "\n"

    # ── 4. Database Schema ───────────────────────────────────────────────
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=5) + "\n\n"

    # ── 5. Category Information ──────────────────────────────────────────
    if predicted_category:
        type_label = _get_category_type_label(predicted_category)
        prompt += "## Diagnosed Category\n"
        prompt += f"**Category:** {predicted_category.get_name()}"
        if predicted_category.get_subname():
            prompt += f" — {predicted_category.get_subname()}"
        prompt += f"\n**Type:** {type_label}\n"
        prompt += f"**Definition:** {predicted_category.get_definition()}\n"

        examples = predicted_category.get_examples()
        if examples:
            prompt += "**Examples:**\n"
            for example in examples:
                prompt += f"  - {example}\n"
        prompt += "\n"

        # Targeted guidance based on category type
        if is_answerable:
            prompt += (
                "Since this is an **Answerable** question, you should generate a SQL query directly. "
                "Do not ask clarification questions — the question is unambiguous and all needed "
                "information is available in the schema and evidence.\n\n"
            )
        elif is_solvable:
            if has_history:
                prompt += (
                    "Since this is an **Ambiguous** question, the user's intent has multiple valid "
                    "interpretations. Review the conversation history: if the user has already clarified "
                    "their intent, generate the SQL that matches their intended interpretation. "
                    "If the ambiguity is still unresolved, ask a follow-up clarification question.\n\n"
                )
            else:
                prompt += (
                    "Since this is an **Ambiguous** question, the user's intent has multiple valid "
                    "interpretations. Ask a clarification question that directly targets the ambiguity "
                    "described by the category above. Do not guess — ask first, then generate SQL "
                    "once the user clarifies.\n\n"
                )
        else:
            prompt += (
                "Since this is an **Unanswerable** question, the database lacks the data or "
                "relationships needed to answer it. Provide feedback that:\n"
                "- Identifies exactly what is missing (specific tables, columns, or external knowledge)\n"
                "- Explains why this prevents answering the question\n"
                "- Suggests what would need to be added to make it answerable\n"
                "Do not attempt to generate SQL or ask clarification questions.\n\n"
            )
    else:
        prompt += (
            "## Taxonomy Reference\n"
            "No category has been diagnosed for this question. Use the taxonomy below to determine "
            "the question type and respond accordingly.\n\n"
        )
        if categories:
            for cat in categories:
                type_label = _get_category_type_label(cat)
                prompt += f"**{cat.get_name()}**"
                if cat.get_subname():
                    prompt += f" — {cat.get_subname()}"
                prompt += f" [{type_label}]\n"
                prompt += f"  {cat.get_definition()}\n"
                examples = cat.get_examples()
                if examples:
                    for example in examples:
                        prompt += f"  - {example}\n"
                prompt += "\n"

    # ── 6. Conversation Constraints ──────────────────────────────────────
    prompt += "## Conversation Status\n"
    prompt += f"**Turn:** {current_step + 1} of {max_steps + 1}\n"
    if is_final_step:
        prompt += (
            "**This is the final turn.** You MUST provide either a SQL query or feedback. "
            "You cannot ask another clarification question.\n"
        )
    else:
        remaining = max_steps - current_step
        prompt += f"You may ask up to {remaining} more clarification question(s).\n"
    prompt += "\n"

    if has_history:
        prompt += (
            "**Important:** Do not repeat questions already asked or request information already provided. "
            "If the user's response was unhelpful, adapt by proceeding with reasonable assumptions "
            "or providing feedback.\n\n"
        )

    # ── 7. Response Options ──────────────────────────────────────────────
    prompt += "## Response Options\n"
    prompt += "Provide exactly ONE of the following:\n\n"

    option_num = 1
    if not is_final_step:
        prompt += (
            f"**Option {option_num}: Clarification Question** (system_question)\n"
            "Use when the question is ambiguous and you need the user to disambiguate. "
            "Your question should be specific, non-technical, and directly target the ambiguity.\n\n"
        )
        option_num += 1

    prompt += (
        f"**Option {option_num}: SQL Query** (system_sql)\n"
        "Use when you have enough information to generate a correct query. "
        "Write valid SQLite using only tables and columns from the schema above.\n\n"
    )
    option_num += 1

    prompt += (
        f"**Option {option_num}: Feedback** (system_feedback)\n"
        "Use when the question cannot be answered with the current database. "
        "Explain what is missing and why.\n\n"
    )

    # ── 8. Output Format ─────────────────────────────────────────────────
    prompt += "## Output Format\n"
    model_for_description = SystemResponseModelLimited if is_final_step else SystemResponseModel
    prompt += model_field_descriptions(model_for_description) + "\n\n"
    prompt += (
        "Provide exactly ONE non-null field. "
        f"The other{'s' if not is_final_step else ''} MUST be null."
    )
    if is_final_step:
        prompt += " system_question is not available on the final turn."
    prompt += "\n"

    model_class = SystemResponseModelLimited if is_final_step else SystemResponseModel
    return prompt, model_class
