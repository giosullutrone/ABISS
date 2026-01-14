from pydantic import Field, BaseModel
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import QuestionUnanswerable


class LexicalVaguenessCategory(Category):
    class LexicalVaguenessOutput(BaseModel):
        reasoning: Annotated[str, Field(description="Use this field to think step-by-step about how to construct the question and SQL queries. First, identify which term in the question lacks a precise boundary or threshold. Second, determine what are the different reasonable interpretations of this vague term. Third, explain how each interpretation affects the SQL query's filtering or selection criteria. Fourth, confirm why the vagueness cannot be resolved from schema information alone. Use this reasoning to guide the generation of the 'question', 'sql_first_interpretation', 'sql_second_interpretation', 'hidden_knowledge_first_interpretation', and 'hidden_knowledge_second_interpretation' fields.")]
        question: Annotated[str, Field(description="A natural language question containing a vague term whose meaning lacks a precise or objective boundary, requiring subjective interpretation for query generation. Examples include temporal expressions (recent, old), quantitative adjectives (many, few, high, low), or evaluative terms (good, popular, expensive).")]
        sql_first_interpretation: Annotated[str, Field(description="The SQL query using the first reasonable interpretation of the vague term (e.g., 'recent' interpreted as within the last month).")]
        sql_second_interpretation: Annotated[str, Field(description="The SQL query using the second reasonable interpretation of the vague term (e.g., 'recent' interpreted as within the last year).")]
        hidden_knowledge_first_interpretation: Annotated[str, Field(description="The hidden user intent clarifying the first interpretation of the vague term (e.g., 'By recent, I mean courses from the last month').")]
        hidden_knowledge_second_interpretation: Annotated[str, Field(description="The hidden user intent clarifying the second interpretation of the vague term (e.g., 'By recent, I mean courses from the last academic year').")]

    @staticmethod
    def get_name() -> str:
        return "Lexical Vagueness"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return "Lexical Vagueness arises when a question contains terms whose meaning lacks a precise or objective boundary, leading to indeterminate selection criteria during query generation. These vague terms require subjective interpretation to establish concrete thresholds or criteria. Such vagueness introduces variability that cannot be resolved solely from schema information, as it depends on the user's subjective understanding or context-specific conventions."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List recent courses.",  # Temporal vagueness: last semester? last year? last month?
            "Show employees with high salaries.",  # Quantitative vagueness: what threshold defines 'high'?
            "Find popular products.",  # Evaluative vagueness: based on sales? reviews? views?
            "Display old buildings on campus.",  # Temporal vagueness: how old is 'old'?
            "Show students with many course enrollments."  # Quantitative vagueness: how many is 'many'?
        ]

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return LexicalVaguenessCategory.LexicalVaguenessOutput

    @staticmethod
    def get_unanswerable_question(db_id: str, output: BaseModel) -> list["QuestionUnanswerable"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, LexicalVaguenessCategory.LexicalVaguenessOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=LexicalVaguenessCategory(),
            question=output.question,
            evidence=None,
            sql=sql,
            hidden_knowledge=hk,
            is_solvable=LexicalVaguenessCategory.is_solvable()
        ) for sql, hk in [
            (output.sql_first_interpretation, output.hidden_knowledge_first_interpretation),
            (output.sql_second_interpretation, output.hidden_knowledge_second_interpretation)
        ]]
