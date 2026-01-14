from pydantic import Field, BaseModel
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class LexicalVaguenessCategory(Category):
    class LexicalVaguenessOutput(BaseModel):
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
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return LexicalVaguenessCategory.LexicalVaguenessOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, LexicalVaguenessCategory.LexicalVaguenessOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=LexicalVaguenessCategory(),
            question=output.question,
            evidence=None,
            sql=sql,
            hidden_knowledge=hk,
            is_solvable=LexicalVaguenessCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        ) for sql, hk in [
            (output.sql_first_interpretation, output.hidden_knowledge_first_interpretation),
            (output.sql_second_interpretation, output.hidden_knowledge_second_interpretation)
        ]]
