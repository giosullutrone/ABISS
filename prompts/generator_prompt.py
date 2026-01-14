from pydantic import BaseModel
from prompts import model_field_descriptions
from db_datasets.db_dataset import DBDataset
from dataset_dataclasses.question import QuestionStyle, QuestionDifficulty
from prompts.generator_style_and_difficulty_prompt import STYLE_DESCRIPTIONS, DIFFICULTY_CRITERIA


def get_generation_prompt(
    db: DBDataset, 
    is_solvable: bool, 
    is_answerable: bool,
    db_id: str, 
    name: str, 
    definition: str, 
    examples: list[str] | None, 
    output: type[BaseModel],
    question_style: QuestionStyle,
    question_difficulty: QuestionDifficulty
) -> str:
    prompt = "You are an expert in creating text-to-SQL benchmarks for natural language questions. " \
             "Your task is to generate realistic, high-quality questions that test the boundaries of text-to-SQL systems.\n\n"
    
    prompt += "## Task Overview\n"
    
    if is_answerable:
        prompt += "Generate natural language questions that:\n" \
                  "1. Can be directly converted to SQL queries without ambiguity or missing information\n" \
                  "2. Clearly map to the database schema with all necessary information available\n" \
                  "3. Are realistic questions a user might actually ask\n" \
                  "4. Have a single, unambiguous interpretation\n" \
                  "5. Fit the specific category defined below\n\n"
    elif is_solvable:
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
              f"{db.get_schema_prompt(db_id, rows=5)}\n\n"
    
    # Add style requirements
    prompt += "## Question Style Requirements\n" \
              f"The question MUST be written in the following style:\n" \
              f"{STYLE_DESCRIPTIONS[question_style]}\n\n"
    
    # Add difficulty requirements
    prompt += "## SQL Difficulty Requirements\n"
    if is_answerable or is_solvable:
        prompt += f"The SQL query (or queries for ambiguous questions) MUST match the following difficulty level:\n" \
                  f"{DIFFICULTY_CRITERIA[question_difficulty]}\n\n"
    else:
        prompt += f"If this question were answerable, the expected SQL query should match the following difficulty level:\n" \
                  f"{DIFFICULTY_CRITERIA[question_difficulty]}\n" \
                  f"Consider this when crafting the question's complexity and scope.\n\n"
    
    prompt += "## Output Format\n" \
              "For each generated question, provide a JSON object with the following fields:\n" \
              f"{model_field_descriptions(output)}\n\n"
    
    prompt += "## Quality Guidelines\n" \
              "Ensure that:\n"
    
    if is_answerable:
        prompt += "- The question is **unambiguous and clear**: There is only one reasonable interpretation\n" \
                  "- The question sounds **natural and realistic**: Something a real user would ask\n" \
                  "- All information needed to answer the question is **available in the schema**\n" \
                  "- The SQL query is **valid and executable** against the provided schema\n" \
                  "- Table and column names from the schema are **used correctly**\n"
    elif is_solvable:
        prompt += "- The ambiguity is **genuine and non-trivial**: Multiple interpretations should be reasonable\n" \
                  "- The question sounds **natural and realistic**: Something a real user would ask\n" \
                  "- The SQL queries for different interpretations are **structurally different**, not just parameter changes\n" \
                  "- The hidden knowledge clearly **resolves the ambiguity** and leads to a specific SQL query\n" \
                  "- All generated SQL is **valid and executable** against the provided schema\n" \
                  "- Table and column names from the schema are **used correctly**\n"
    else:
        prompt += "- The question is **well-formed and natural**: Something a real user would reasonably ask\n" \
                  "- The question **clearly cannot be answered** with the current schema\n" \
                  "- It's obvious **what is missing** from the schema (specific tables, columns, or relationships)\n" \
                  "- The question doesn't rely on data values but on **structural schema limitations**\n" \
                  "- Avoid questions that are answerable with existing schema elements through creative joins or aggregations\n" \
                  "- The missing elements should be **realistic additions** that would logically belong in such a database\n"
    prompt += "- The SQL complexity **matches the specified difficulty level**\n" \
              "- The question style **matches the specified style requirements**\n"
    
    prompt += "\n## Generation Process\n" \
              "Follow this thinking process when generating the question:\n"
    
    if is_answerable:
        prompt += "1. **First**, design the SQL query that matches the specified difficulty level and correctly uses the database schema\n" \
                  "2. **Then**, craft a natural language question in the specified style that unambiguously maps to this SQL query\n" \
                  "3. **Finally**, ensure all output fields are complete and consistent\n"
    elif is_solvable:
        prompt += "1. **First**, design the SQL query (or queries if multiple interpretations exist) that matches the specified difficulty level\n" \
                  "2. **Then**, craft a natural language question in the specified style that requires clarification or additional user knowledge\n" \
                  "3. **Next**, formulate the hidden knowledge that would resolve the ambiguity or provide the missing information\n" \
                  "4. **Finally**, ensure all output fields are complete and consistent\n"
    else:
        prompt += "1. **First**, think about what SQL query would be needed if the schema were complete (matching the specified difficulty level)\n" \
                  "2. **Then**, identify what specific schema elements (tables, columns, relationships) are missing\n" \
                  "3. **Next**, craft a natural language question in the specified style that would require these missing elements\n" \
                  "4. **Finally**, ensure the question clearly cannot be answered with the current schema\n"
    
    prompt += "\nGenerate the question sample now."
    return prompt
