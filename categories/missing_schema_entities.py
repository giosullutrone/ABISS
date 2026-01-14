from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class MissingSchemaEntitiesCategory(Category):
    class MissingSchemaEntitiesOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question that requests information about entities or attributes that do not exist in the database schema. The question should be well-formed and realistic, but impossible to answer because key tables or columns are completely absent from the schema.")]

    @staticmethod
    def get_name() -> str:
        return "Missing Schema Elements"

    @staticmethod
    def get_subname() -> str | None:
        return "Missing Entities or Attributes"

    @staticmethod
    def get_definition() -> str:
        return "A question is unanswerable due to Missing Entities or Attributes when key information is completely absent from the database schema. This occurs when the schema contains no table or column that could represent the requested data, making it impossible to construct any SQL query to answer the question. The missing elements are fundamental entities or attributes that would need to be added to the schema structure itself."

    @staticmethod
    def get_examples() -> list[str] | None:
        return None
    
    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return False

    @staticmethod
    def get_output() -> type[BaseModel]:
        return MissingSchemaEntitiesCategory.MissingSchemaEntitiesOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, MissingSchemaEntitiesCategory.MissingSchemaEntitiesOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=MissingSchemaEntitiesCategory(),
            question=output.question,
            evidence=None,
            sql=None,
            hidden_knowledge=None,
            is_solvable=MissingSchemaEntitiesCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        )]
