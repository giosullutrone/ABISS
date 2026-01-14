from generators import model_field_descriptions
from categories.category import Category
from dataset_dataclasses.question import QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from typing import Literal


class CategoryCheckResponse(BaseModel):
    thinking_process: Annotated[str, Field(description="Step-by-step analysis of whether the question fits the specified category. Consider: " \
    "(1) the category's definition and characteristics, "
    "(2) the nature of the ambiguity or unanswerability in the question, "
    "(3) whether the question exhibits the specific issues defined by the category, and "
    "(4) for solvable categories, whether the hidden knowledge appropriately resolves the ambiguity.")]
    answer: Annotated[Literal["Yes", "No"], Field(description="Final verdict: 'Yes' if the question correctly belongs to the specified category, or "
    "'No' if it doesn't match the category's characteristics or belongs to a different category.")]

def get_category_validation_result(response: str) -> bool:
    try:
        response_json = CategoryCheckResponse.model_validate_json(response)
        answer = response_json.answer.strip().lower()
        return "yes" in answer
    except Exception:
        return False

def get_category_validation_prompt(db: DBDataset, category: Category, question: QuestionUnanswerable) -> str:
    prompt = "You are an expert in Text-to-SQL ambiguity and unanswerability classification. " \
                "Your task is to verify whether a generated question correctly belongs to a specific category " \
                "based on the category's definition and characteristics.\n\n"
    
    prompt += "## Category Information\n"
    prompt += f"**Category Name:** {category.get_name()}\n"
    if category.get_subname():
        prompt += f"**Subcategory:** {category.get_subname()}\n"
    prompt += f"**Category Definition:** {category.get_definition()}\n"
    
    examples = category.get_examples()
    if examples:
        prompt += "**Category Examples:**\n"
        for example in examples:
            prompt += f"  - {example}\n"
    prompt += "\n"
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=5, db_sql_manipulation=None) + "\n\n"
    
    prompt += "## Question to Validate\n"
    prompt += f"**Natural Language Question:** {question.question}\n"
    
    if question.is_solvable:
        # Solvable category - should have hidden knowledge and SQL
        if question.hidden_knowledge:
            prompt += f"**Hidden Knowledge:** {question.hidden_knowledge}\n"
        
        if question.sql:
            prompt += f"**Ground Truth SQL:** {question.sql}\n"
            
            results = db.execute_query(
                db_id=question.db_id, 
                sql_query=question.sql, 
                db_sql_manipulation=None
            )
            if results is not None:
                prompt += f"**Query Results (first 5 rows):** {results[:5]}\n"
    
    prompt += "\n## Validation Task\n"
    
    if question.is_solvable:
        prompt += "For this **solvable** category, verify that:\n"
        prompt += "1. The question exhibits the specific type of ambiguity or underspecification defined by the category\n"
        prompt += "2. The question cannot be directly converted to SQL without the hidden knowledge\n"
        prompt += "3. The hidden knowledge appropriately resolves the ambiguity as per the category's characteristics\n"
        prompt += "4. With the hidden knowledge, the question has a clear, unambiguous answer\n"
        prompt += "5. The provided SQL correctly implements the disambiguated intent\n"
        prompt += "6. The question is realistic and natural for this type of ambiguity\n\n"
        
        prompt += "Answer **Yes** if all these conditions are met and the question genuinely belongs to this category.\n"
        prompt += "Answer **No** if:\n"
        prompt += "- The question doesn't exhibit the category's specific characteristics\n"
        prompt += "- The ambiguity can be resolved without the provided hidden knowledge\n"
        prompt += "- The hidden knowledge doesn't properly resolve the ambiguity\n"
        prompt += "- The question would fit better in a different category\n"
    else:
        prompt += "For this **unsolvable** category, verify that:\n"
        prompt += "1. The question exhibits the specific type of unanswerability defined by the category\n"
        prompt += "2. The question is well-formed and appears to be a reasonable query\n"
        prompt += "3. The database schema genuinely lacks the elements needed to answer the question\n"
        prompt += "4. The question cannot be answered through creative SQL queries or workarounds\n"
        prompt += "5. The missing schema elements align with the category's definition\n"
        prompt += "6. The question is realistic - something a user might actually ask\n\n"
        
        prompt += "Answer **Yes** if all these conditions are met and the question genuinely belongs to this category.\n"
        prompt += "Answer **No** if:\n"
        prompt += "- The question doesn't exhibit the category's specific characteristics\n"
        prompt += "- The question IS actually answerable with the current schema\n"
        prompt += "- The issue is not a schema limitation but another type of problem\n"
        prompt += "- The question would fit better in a different category\n"
    
    prompt += "\n## Response Format\n"
    prompt += "Provide your evaluation as a JSON object with:\n"
    prompt += model_field_descriptions(CategoryCheckResponse) + "\n\n"
    
    prompt += "Carefully analyze whether the question truly belongs to this specific category before deciding."
    return prompt
