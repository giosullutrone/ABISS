from prompts import model_field_descriptions
from categories.category import Category
from dataset_dataclasses.question import Question, QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from typing import Literal


class CategoryCheckResponse(BaseModel):
    answer: Annotated[Literal["Yes", "No"], Field(description="Final verdict: 'Yes' if the question correctly belongs to the specified category, or "
    "'No' if it doesn't match the category's characteristics or belongs to a different category. Put only 'Yes' or 'No'.")]

def get_category_validation_result(response: BaseModel) -> bool:
    answer = CategoryCheckResponse.model_validate(response).answer.strip().lower()
    return "yes" in answer

def get_category_validation_prompt(db: DBDataset, category: Category, question: Question) -> str:
    prompt = "You are an expert in text-to-SQL ambiguity and unanswerability classification. " \
             "Your task is to verify whether a generated question correctly belongs to a specific category " \
             "based on the category's definition, characteristics, and provided examples.\n\n"
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=5) + "\n\n"
    
    prompt += "## Category Information\n"
    prompt += f"**Category Name:** {category.get_name()}"
    if category.get_subname():
        prompt += f" - {category.get_subname()}"
    prompt += "\n"
    prompt += f"**Definition:** {category.get_definition()}\n\n"
    
    examples = category.get_examples()
    if examples:
        prompt += "**Category Examples:**\n"
        for example in examples:
            prompt += f"  - {example}\n"
        prompt += "\n"
    
    prompt += "## Question to Validate\n"
    prompt += f"**Natural Language Question:** {question.question}\n"
    
    # Include SQL for all questions (both answerable and unanswerable)
    if question.sql:
        prompt += f"**Ground Truth SQL:** {question.sql}\n"
        
        results = db.execute_query(
            db_id=question.db_id, 
            sql_query=question.sql
        )
        if results is not None:
            prompt += f"**Query Results (first 5 rows):** {results[:5]}\n"
    
    # Only include hidden knowledge if the question is of type QuestionUnanswerable
    if isinstance(question, QuestionUnanswerable):
        if question.hidden_knowledge:
            prompt += f"**Hidden Knowledge:** {question.hidden_knowledge}\n"
    
    prompt += "\n## Validation Task\n"
    
    if isinstance(question, QuestionUnanswerable):
        if question.is_solvable:
            prompt += "For this **solvable** category, verify that:\n"
            prompt += "1. The question exhibits the specific type of ambiguity or underspecification defined by the category\n"
            prompt += "2. The question cannot be directly converted to SQL without the hidden knowledge\n"
            prompt += "3. The hidden knowledge appropriately resolves the ambiguity according to the category's characteristics\n"
            prompt += "4. With the hidden knowledge, the question has a clear, unambiguous answer\n"
            prompt += "5. The provided SQL correctly implements the disambiguated intent\n"
            prompt += "6. The question is realistic and natural for this type of ambiguity\n\n"
            
            prompt += "**Answer 'Yes' if:** All conditions are met and the question genuinely belongs to this category.\n\n"
            prompt += "**Answer 'No' if:**\n"
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
            prompt += "6. The question is realistic and something a user might actually ask\n\n"
            
            prompt += "**Answer 'Yes' if:** All conditions are met and the question genuinely belongs to this category.\n\n"
            prompt += "**Answer 'No' if:**\n"
            prompt += "- The question doesn't exhibit the category's specific characteristics\n"
            prompt += "- The question is actually answerable with the current schema\n"
            prompt += "- The issue is not a schema limitation but another type of problem\n"
            prompt += "- The question would fit better in a different category\n"
    else:
        # Regular answerable question
        prompt += "For this **answerable** category, verify that:\n"
        prompt += "1. The question is clear and unambiguous\n"
        prompt += "2. The question can be directly answered using the available database schema\n"
        prompt += "3. The question does not require any hidden knowledge, external information, or disambiguation\n"
        prompt += "4. The provided SQL query is syntactically correct and executable\n"
        prompt += "5. The SQL query correctly answers the question as stated\n"
        prompt += "6. The question does not exhibit any ambiguity or characteristics of other categories\n"
        prompt += "7. The question is realistic and natural\n\n"
        
        prompt += "**Answer 'Yes' if:** All conditions are met and the question is genuinely answerable without ambiguity.\n\n"
        prompt += "**Answer 'No' if:**\n"
        prompt += "- The question is ambiguous or exhibits characteristics of other categories (vagueness, ambiguity, etc.)\n"
        prompt += "- The question requires information not available in the database\n"
        prompt += "- The SQL query doesn't correctly answer the question\n"
        prompt += "- The SQL query has syntax errors or cannot be executed\n"
        prompt += "- The question would fit better in a different category\n"
    
    prompt += "\n## Response Format\n"
    prompt += "Provide a concise but thorough analysis evaluating whether the question fits the specified category. " \
              "Your reasoning should systematically address the validation criteria listed above, focusing on: " \
              "(1) how the question aligns with the category's core definition, " \
              "(2) whether the question exhibits the specific characteristics unique to this category, " \
              "(3) comparison with the provided category examples, and " \
              "(4) any evidence that suggests the question might belong to a different category instead.\n\n"
    prompt += "Then provide your final verdict as a JSON object with:\n"
    prompt += model_field_descriptions(CategoryCheckResponse) + "\n\n"
    prompt += "Remember: Be strict and objective. The question should clearly and unambiguously exhibit the category's defining characteristics."
    
    return prompt
