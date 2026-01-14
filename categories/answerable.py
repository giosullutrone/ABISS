from pydantic import Field, BaseModel
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question


class AnswerableCategory(Category):
    class AnswerableOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question that can be directly answered using the database schema and available data without any ambiguity or missing information.")]
        sql: Annotated[str, Field(description="The SQL query that correctly answers the question based on the database schema.")]

    @staticmethod
    def get_name() -> str:
        return "Answerable"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return "A question is Answerable when it can be directly answered using a SQL query against the database without requiring additional information, clarification, or disambiguation. The question clearly maps to the database schema, and all necessary information to construct a correct SQL query is available."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "How many students are enrolled?",
            "List all courses in the Computer Science department.",
            "What is the average grade for students in Biology 101?",
            "Show the names of professors who teach in the Engineering department."
        ]

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return AnswerableCategory.AnswerableOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel) -> list["Question"]:
        from dataset_dataclasses.question import Question
        assert isinstance(output, AnswerableCategory.AnswerableOutput)
        return [Question(
            db_id=db_id,
            category=AnswerableCategory(),
            question=output.question,
            evidence=None,
            sql=output.sql
        )]
