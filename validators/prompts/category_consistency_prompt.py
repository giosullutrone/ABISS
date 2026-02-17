from utils.prompt_utils import model_field_descriptions
from categories.category import Category
from dataset_dataclasses.question import Question, QuestionUnanswerable
from db_datasets.db_dataset import DBDataset
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field
from typing import Literal


class CategoryConsistencyResponse(BaseModel):
    answer: Annotated[Literal["A", "B"], Field(description="Final selection: 'A' if the question better fits Category A, 'B' if it better fits Category B. Put only 'A' or 'B'.")]

def get_category_consistency_result(response: BaseModel) -> int:
    """Parses the model response and returns 0 if Category A is better, 1 if Category B is better."""
    answer = CategoryConsistencyResponse.model_validate(response).answer.strip().upper()
    if "A" in answer:
        return 0
    elif "B" in answer:
        return 1
    raise ValueError("Invalid answer in CategoryConsistencyResponse: must contain 'A' or 'B'.")

def get_category_consistency_prompt(db: DBDataset, category_a: Category, category_b: Category, question: Question) -> str:
    is_ambiguous = isinstance(question, QuestionUnanswerable) and question.is_solvable

    prompt = "You are an expert in text-to-SQL ambiguity and unanswerability classification. " \
             "Your task is to perform a comparative analysis between two categories and determine which one " \
             "a question more closely aligns with based on their definitions, characteristics, and examples.\n\n"
    
    prompt += "## Database Schema\n"
    prompt += db.get_schema_prompt(question.db_id, rows=5) + "\n\n"
    
    prompt += "## Question Under Analysis\n"
    prompt += f"**Natural Language Question:** {question.question}\n"
    
    if question.evidence:
        prompt += f"**Evidence/Context:** {question.evidence}\n"
    
    # Include SQL for all questions (both answerable and unanswerable)
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
    
    prompt += "\n## Category A\n"
    prompt += f"**Name:** {category_a.get_name()}"
    if category_a.get_subname():
        prompt += f" - {category_a.get_subname()}"
    prompt += "\n"
    prompt += f"**Definition:** {category_a.get_definition()}\n"
    
    examples_a = category_a.get_examples()
    if examples_a:
        prompt += "**Examples:**\n"
        for example in examples_a:
            prompt += f"  - {example}\n"
    prompt += "\n"
    
    prompt += "## Category B\n"
    prompt += f"**Name:** {category_b.get_name()}"
    if category_b.get_subname():
        prompt += f" - {category_b.get_subname()}"
    prompt += "\n"
    prompt += f"**Definition:** {category_b.get_definition()}\n"
    
    examples_b = category_b.get_examples()
    if examples_b:
        prompt += "**Examples:**\n"
        for example in examples_b:
            prompt += f"  - {example}\n"
    prompt += "\n"
    
    prompt += "## Comparative Evaluation Task\n"
    prompt += "You must select which category the question fits BETTER. This is a forced choice between Category A and Category B. " \
              "Even if the question partially fits both or neither perfectly, you must determine which category's " \
              "core characteristics and defining features the question more strongly exhibits.\n\n"
    
    prompt += "**Evaluation Criteria (in order of importance):**\n"
    prompt += "1. **Core Definition Alignment**: Which category's fundamental definition the question best exemplifies\n"
    prompt += "2. **Distinctive Characteristics**: Which category's unique/specific features are present in the question\n"
    prompt += "3. **Problem Type Match**: Whether the underlying issue matches one category's problem type better\n"

    # Add type-specific guidance based on whether the question is ambiguous or unanswerable
    if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
        if is_ambiguous:
            prompt += "4. **Disambiguation Alignment**: Which category's type of ambiguity the disambiguation information better resolves (i.e., the disambiguation should match the nature of the ambiguity described by the category)\n"
        else:
            prompt += "4. **Feedback Alignment**: Which category's unsolvability cause the feedback better describes (i.e., the feedback should match the specific limitation described by the category)\n"

    has_extra_criterion = isinstance(question, QuestionUnanswerable) and question.hidden_knowledge
    prompt += f"{'5' if has_extra_criterion else '4'}. **Example Similarity**: Which category's examples are most similar to the question\n"
    
    prompt += "\n**Select 'A' if:** Category A is the better fit based on the criteria above.\n"
    prompt += "**Select 'B' if:** Category B is the better fit based on the criteria above.\n\n"
    
    prompt += "## Response Format\n"
    prompt += "Provide a structured comparative analysis with the following sections:\n"
    prompt += "1. **Question Analysis**: Briefly identify the key characteristics of the question\n"
    prompt += "2. **Category A Match**: Evaluate how well the question aligns with Category A (consider definition, characteristics, examples)\n"
    prompt += "3. **Category B Match**: Evaluate how well the question aligns with Category B (consider definition, characteristics, examples)\n"
    prompt += "4. **Final Determination**: Clearly state which category is the better fit and why\n\n"
    prompt += "Keep your reasoning concise but thorough (approximately 200-400 words total). " \
              "Be decisive and objective in your comparison.\n\n"
    prompt += "Then provide your final selection as a JSON object with:\n"
    prompt += model_field_descriptions(CategoryConsistencyResponse) + "\n\n"
    prompt += "Important: Choose the category that most closely matches the question's primary characteristics. " \
              "If the question exhibits features of both categories, select the one with the STRONGEST alignment."
    
    return prompt
