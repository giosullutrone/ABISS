from pydantic import BaseModel
from utils.prompt_utils import model_field_descriptions
from db_datasets.db_dataset import DBDataset
from dataset_dataclasses.question import QuestionStyle, QuestionDifficulty
from utils.style_and_difficulty_utils import STYLE_DESCRIPTIONS_WITH_QUESTION_EXAMPLES, DIFFICULTY_CRITERIA


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
                  "3. Are realistic questions a user might actually ask on the given database and evidence (if available)\n" \
                  "4. Have a single, unambiguous interpretation\n" 
    elif is_solvable:
        prompt += "Generate natural language questions that:\n" \
                  "1. Are genuinely ambiguous and have multiple valid interpretations\n" \
                  "2. Cannot be converted to SQL queries without additional clarification from the user\n" \
                  "3. Are realistic questions a user might actually ask on the given database\n" \
                  "4. Can be resolved by using the hidden knowledge which represents the disambiguation information\n" 
    else:
        prompt += "Generate natural language questions that:\n" \
                  "1. Appear to be valid questions a user might ask\n" \
                  "2. Cannot be answered with either the current database schema, evidence, or both\n" \
                  "3. Are realistic and well-formed questions\n" \
                  "4. Are not trivially solvable or easily answerable\n" 
    
    prompt += "5. Fit the specific category defined below\n\n"
    
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
              f"{STYLE_DESCRIPTIONS_WITH_QUESTION_EXAMPLES[question_style]}\n\n"
    
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
                  "- All information needed to answer the question is **available in the schema or evidence (if available)**\n" \
                  "- The SQL query is **valid and executable** against the provided schema\n" 
    elif is_solvable:
        prompt += "- The ambiguity is **genuine and non-trivial**: Multiple interpretations should be reasonable\n" \
                  "- The question sounds **natural and realistic**: Something a real user would ask\n" \
                  "- The SQL queries for different interpretations are **structurally different**, not just parameter changes\n" \
                  "- The hidden knowledge clearly **resolves the ambiguity** and leads to a specific SQL query\n" \
                  "- All generated SQL is **valid and executable** against the provided schema\n" 
    else:
        prompt += "- The question is **well-formed and natural**: Something a real user would reasonably ask\n" \
                  "- The question **clearly cannot be answered** with the current schema\n" \
                  "- Avoid questions that are answerable with existing schema elements through creative joins or aggregations\n"
    
    prompt += "- The SQL complexity **matches the specified difficulty level**\n" \
              "- The question style **matches the specified style requirements**\n"
    
    prompt += "\n## Generation Process\n" \
              "Develop your question through an iterative refinement process, moving back and forth between the output fields as needed to ensure coherence and quality. " \
              "Consider the output schema fields in the order they are defined, but feel free to revise earlier fields based on insights from later ones:\n\n"
    
    if is_answerable:
        prompt += "- **question**: Craft a natural language question that fits the category and style\n" \
                  "- **sql**: Develop the SQL query that answers the question with the specified difficulty\n" \
                  "- Iterate between these fields to ensure the question clearly and unambiguously maps to the SQL query\n"
    elif is_solvable:
        prompt += "- **question**: Craft a natural language question that fits the category and style\n" \
                  "- **hidden_knowledge**: Formulate the information that resolves the ambiguity or provides missing context\n" \
                  "- **sql_with_user_knowledge** (or equivalent): Develop the SQL query that answers the question given the hidden knowledge\n" \
                  "- Iterate between these fields to ensure the question genuinely requires the hidden knowledge and the SQL correctly incorporates it\n"
    else:
        prompt += "- **question**: Craft a natural language question that fits the category and style\n" \
                  "- **feedback**: Articulate why this question cannot be answered with the current schema\n" \
                  "- Iterate between these fields to ensure the question clearly requires missing schema elements and the feedback precisely explains what is missing\n"
    
    prompt += "\nGenerate the question sample now."
    return prompt
