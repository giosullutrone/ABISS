from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class MissingSchemaRelationshipsCategory(Category):
    class MissingSchemaRelationshipsOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question that requires connecting two or more entities that exist in the schema but lack any relationship or foreign key linkage between them. The question should be well-formed and realistic, appearing answerable at first glance because the relevant tables exist, but impossible to answer because no connection exists between them.")]
        feedback: Annotated[str, Field(description="Explanation of why this question is not solvable, specifying which entities exist in the schema but lack the necessary relationships or foreign keys to connect them and answer the question.")]

    @staticmethod
    def get_name() -> str:
        return "Missing Schema Elements"

    @staticmethod
    def get_subname() -> str | None:
        return "Missing Relationships"

    @staticmethod
    def get_definition() -> str:
        return "A question is unanswerable due to Missing Relationships when no linkage exists between relevant entities in the database schema. This occurs when the schema includes the necessary tables for the entities mentioned in the question, but lacks the foreign keys, junction tables, or other relationship structures needed to connect them. Without these relationships, it is impossible to construct a SQL query that joins the relevant information together to answer the question."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the professors for the course 'database'. (Unanswerable: both teachers and courses tables exist, but no foreign key or junction table links them)",
            "Show which students are advised by which professors. (Unanswerable: students and professors tables exist, but no advising relationship connects them)",
            "Find the suppliers for products in the electronics category. (Unanswerable: suppliers and products tables exist, but no linkage table relates them)",
            "What buildings are each department located in? (Unanswerable: departments and buildings tables exist, but no assignment relationship links them)"
        ]

    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return False

    @staticmethod
    def get_output() -> type[BaseModel]:
        return MissingSchemaRelationshipsCategory.MissingSchemaRelationshipsOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, MissingSchemaRelationshipsCategory.MissingSchemaRelationshipsOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=MissingSchemaRelationshipsCategory(),
            question=output.question,
            evidence=None,
            sql=None,
            hidden_knowledge=output.feedback,
            is_solvable=MissingSchemaRelationshipsCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        )]