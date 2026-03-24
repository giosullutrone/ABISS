from db_datasets.sql_generation_prompts import SQLGenerationResponse, get_sql_generation_prompt

# Re-export for use by the generator's Phase 2
SQLGenerationOutput = SQLGenerationResponse

__all__ = ["SQLGenerationOutput", "get_sql_generation_prompt"]
