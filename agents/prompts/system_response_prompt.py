from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel, Field
from typing import Annotated
from utils.prompt_utils import model_field_descriptions, get_conversation_history_prompt
from dataset_dataclasses.benchmark import Conversation, SystemResponse
from categories.category import Category


class SystemResponseModel(BaseModel):
    system_question: Annotated[str | None, Field(description="A clear, concise clarification question to ask the user that will help resolve ambiguity or obtain missing information needed to generate a SQL query. Should be None if you're providing SQL or feedback instead.")]
    system_sql: Annotated[str | None, Field(description="A valid SQLite query that correctly answers the user's question based on the database schema. Should be None if you're asking a question or providing feedback instead.")]
    system_feedback: Annotated[str | None, Field(description="A clear, helpful explanation of why the question cannot be answered with the current database schema and what would be needed to answer it. Should be None if you're asking a question or providing SQL instead.")]

class SystemResponseModelLimited(BaseModel):
    system_sql: Annotated[str | None, Field(description="A valid SQLite query that correctly answers the user's question based on the database schema. Should be None if you're providing feedback instead.")]
    system_feedback: Annotated[str | None, Field(description="A clear, helpful explanation of why the question cannot be answered with the current database schema and what would be needed to answer it. Should be None if you're providing SQL instead.")]


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

def get_system_response_prompt(
    db: DBDataset, 
    conversation: Conversation, 
    predicted_category: Category | None,
    current_step: int,
    max_steps: int
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
    
    Returns:
        A tuple of (prompt, model_class) where model_class is the appropriate Pydantic model
        to use for constrained generation.
    """
    question = conversation.question
    has_history = len(conversation.interactions) > 0
    is_answerable = predicted_category.is_answerable() if predicted_category else None
    is_solvable = predicted_category.is_solvable() if predicted_category else None
    is_final_step = current_step >= max_steps
    
    prompt = (
        "You are an expert text-to-SQL system. Based on the user's question and the database schema, "
        "you need to provide exactly ONE of the following responses:\n"
        "1. **Clarification Question** - If the question is ambiguous or lacks information\n"
        "2. **SQL Query** - If you can generate a correct SQL query to answer the question\n"
        "3. **Feedback** - If the question is fundamentally unanswerable with the current database\n\n"
    )
    
    # Add step information
    prompt += f"**Current Step:** {current_step + 1}/{max_steps + 1}\n"
    if is_final_step:
        prompt += (
            f"**IMPORTANT CONSTRAINT:** This is step {max_steps + 1} (the final step). You have asked {max_steps} clarification question(s). "
            "You MUST now provide either a SQL query (if you can) or feedback (if the question is truly unanswerable). "
            "You CANNOT ask another clarification question.\n\n"
        )
    else:
        remaining_questions = max_steps - current_step
        prompt += f"You can ask up to {remaining_questions} more clarification question(s) before providing a final response.\n\n"
    
    prompt += "## Category Analysis\n"
    if predicted_category:
        prompt += f"You classified this question as: **{predicted_category.get_name()}**"
        if predicted_category.get_subname():
            prompt += f" ({predicted_category.get_subname()})"
        prompt += f"\n\n**Definition:** {predicted_category.get_definition()}\n"
        
        examples = predicted_category.get_examples()
        if examples:
            prompt += f"\n**Examples of {predicted_category.get_name()}:**\n"
            for i, example in enumerate(examples, 1):
                prompt += f"{i}. {example}\n"
        prompt += "\n"
        
        # Provide guidance based on category
        if is_answerable:
            prompt += "This is an **answerable** question, so you should provide a SQL query.\n\n"
        elif is_solvable:
            if is_final_step:
                prompt += (
                    "This question may be solvable with clarification, but you've reached the final step. "
                    "Provide your best attempt at a SQL query or explain why it's impossible.\n\n"
                )
            else:
                prompt += (
                    "This question may be solvable with clarification. Ask a focused question to resolve "
                    "the ambiguity, or provide SQL if you believe you now have enough information.\n\n"
                )
        else:
            prompt += (
                "This is an **unsolvable** question with the current schema. You should provide feedback "
                "explaining why it cannot be answered.\n\n"
            )
    else:
        prompt += (
            "No category has been predicted for this question yet. Analyze the question to determine "
            "whether it is answerable, requires clarification, or is fundamentally unsolvable.\n\n"
        )
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=5) + "\n\n"
    
    # Add conversation history or original question
    if has_history:
        prompt += "## Conversation History\n"
        prompt += get_conversation_history_prompt(conversation)
        prompt += "\n"
    else:
        prompt += "## Original Question\n"
        prompt += f"**User's Question:** {question.question}\n"
        if question.evidence:
            prompt += f"**Additional Context:** {question.evidence}\n"
        prompt += "\n"
    
    # Decision guidance
    prompt += "## Your Task\n"
    prompt += "Analyze the question and conversation history (if any), then provide exactly ONE response:\n\n"
    
    if not is_final_step:
        prompt += (
            "### Option 1: Clarification Question (system_question)\n"
            "Choose this if:\n"
            "- The question is ambiguous or underspecified\n"
            "- You need specific information from the user to generate correct SQL\n"
            "- The ambiguity relates to natural language interpretation (e.g., unclear entities, vague terms)\n"
            f"- You have not yet used all {max_steps} allowed clarification question(s)\n\n"
            "Your question should:\n"
            "- Directly address the specific ambiguity or missing information\n"
            "- Be focused and answerable concisely\n"
            "- Help you generate correct SQL once answered\n"
            "- Avoid technical jargon or SQL-specific details\n\n"
        )
    
    prompt += f"### Option {2 if not is_final_step else 1}: SQL Query (system_sql)\n"
    prompt += "Choose this if:\n"
    prompt += "- You have all information needed to generate a correct SQL query\n"
    if has_history:
        prompt += "- The user's clarifications resolved any ambiguities\n"
    prompt += "- The question maps clearly to the database schema\n"
    if is_final_step:
        prompt += "- You can make reasonable assumptions to answer the question\n"
    prompt += (
        "\n"
        "Your SQL should:\n"
        "- Use valid SQLite syntax\n"
        "- Correctly use tables and columns from the schema\n"
        "- Apply appropriate joins, filters, and aggregations\n"
        "- Return exactly what the question asks for\n"
    )
    if has_history:
        prompt += "- Incorporate all clarifications from the conversation\n"
    prompt += "\n"
    
    prompt += f"### Option {3 if not is_final_step else 2}: Feedback (system_feedback)\n"
    prompt += "Choose this if:\n"
    prompt += "- The question requires data not in the database schema\n"
    prompt += "- Required relationships or entities are missing from the schema\n"
    prompt += "- The question requires external knowledge not stored in the database\n"
    prompt += "- The question is fundamentally malformed or doesn't make sense\n"
    if is_final_step:
        prompt += f"- Even after {max_steps} clarification attempt(s), the question remains unanswerable\n"
    prompt += (
        "\n"
        "Your feedback should:\n"
        "- Clearly explain why the question cannot be answered\n"
        "- Identify specific missing schema elements or data\n"
        "- Suggest what would need to be added to answer the question\n"
        "- Be polite and constructive\n\n"
    )
    
    prompt += "## Output Format\n"
    prompt += "Provide your response as a JSON object with these fields:\n"
    
    # Use the appropriate model description based on whether questions are still allowed
    model_for_description = SystemResponseModelLimited if is_final_step else SystemResponseModel
    prompt += model_field_descriptions(model_for_description) + "\n\n"
    
    prompt += (
        f"**CRITICAL:** Provide exactly ONE non-null field. "
        f"The other{' MUST be null.' if is_final_step else ' two MUST be null.'}\n"
    )
    if is_final_step:
        prompt += f"**REMINDER:** system_question is not available because you are at the final step ({max_steps + 1}/{max_steps + 1}).\n"
    prompt += "\nGenerate your response now."
    
    # Return the appropriate model class based on whether questions are still allowed
    model_class = SystemResponseModelLimited if is_final_step else SystemResponseModel
    return prompt, model_class
