from pydantic import BaseModel
from prompts import model_field_descriptions
from db_datasets.db_dataset import DBDataset


def get_generation_prompt(db: DBDataset, is_solvable: bool, db_id: str, name: str, definition: str, examples: list[str] | None, output: type[BaseModel]) -> str:
    prompt = f"You are an expert in creating Text-to-SQL benchmarks for natural language questions. " \
            "Your task is to generate realistic, high-quality questions that test the boundaries of Text-to-SQL systems.\n\n"
    
    prompt += "## Task Overview\n"
    
    if is_solvable:
        prompt += "Generate natural language questions that:\n" \
                "1. Are genuinely ambiguous and have multiple valid interpretations\n" \
                "2. Cannot be converted to SQL queries without additional clarification from the user\n" \
                "3. Are realistic questions a user might actually ask\n" \
                "4. Can be resolved through user interaction to obtain hidden knowledge\n" \
                "5. Fit the specific category defined below\n\n"
    else:
        prompt += "Generate natural language questions that:\n" \
                "1. Appear to be valid questions a user might ask\n" \
                "2. Cannot be answered with the current database schema\n" \
                "3. Are realistic and well-formed questions\n" \
                "4. Would require schema modifications (new tables, columns, or relationships) to answer\n" \
                "5. Fit the specific category defined below\n\n"
    
    prompt += "## Category Details\n" \
            f"**Category Name:** {name}\n" \
            f"**Definition:** {definition}\n\n"
    
    if examples:
        prompt += "## Category Examples\n" \
                "Here are illustrative examples of questions in this category:\n"
        for example in examples:
            prompt += f"- {example}\n"
        prompt += "\n"
    
    prompt += "## Database Schema\n" \
            "Generate questions based on the following database schema:\n\n" \
            f"{db.get_schema_prompt(db_id, rows=5, db_sql_manipulation=None)}\n\n"
    
    prompt += "## Output Format\n" \
            "For each generated question, provide a JSON object with the following fields:\n" \
            f"{model_field_descriptions(output)}\n\n"
    
    prompt += "## Quality Guidelines\n" \
            "Ensure that:\n"
    
    if is_solvable:
        prompt += "- The ambiguity is **genuine and non-trivial** - multiple interpretations should be reasonable\n" \
                "- The question sounds **natural and realistic** - something a real user would ask\n" \
                "- The SQL queries for different interpretations are **structurally different**, not just parameter changes\n" \
                "- The hidden knowledge clearly **resolves the ambiguity** and leads to a specific SQL query\n" \
                "- All generated SQL is **valid and executable** against the provided schema\n" \
                "- Table and column names from the schema are **used correctly**\n"
    else:
        prompt += "- The question is **well-formed and natural** - something a real user would reasonably ask\n" \
                "- The question **clearly cannot be answered** with the current schema\n" \
                "- It's obvious **what is missing** from the schema (specific tables, columns, or relationships)\n" \
                "- The question doesn't rely on data values but on **structural schema limitations**\n" \
                "- Avoid questions that are answerable with existing schema elements through creative joins or aggregations\n" \
                "- The missing elements should be **realistic additions** that would logically belong in such a database\n"
    
    prompt += "\nGenerate the question samples now."
    return prompt
