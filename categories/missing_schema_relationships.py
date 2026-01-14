from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import QuestionUnanswerable


class MissingSchemaRelationshipsCategory(Category):
    class MissingSchemaRelationshipsOutput(BaseModel):
        reasoning: Annotated[str, Field(description="Use this field to think step-by-step about how to construct the question. First, determine what relationship the question should require between entities. Second, identify which tables exist in the schema that seem relevant. Third, consider what foreign keys or junction tables would be needed to link them. Fourth, explain why the absence of these relationships makes the question fundamentally unanswerable without schema modifications. Use this reasoning to guide the generation of the 'question' field.")]
        question: Annotated[str, Field(description="A natural language question that requires connecting two or more entities that exist in the schema but lack any relationship or foreign key linkage between them. The question should be well-formed and realistic, appearing answerable at first glance because the relevant tables exist, but impossible to answer because no connection exists between them.")]

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
        return None

    @staticmethod
    def is_solvable() -> bool:
        return False

    @staticmethod
    def get_output() -> type[BaseModel]:
        return MissingSchemaRelationshipsCategory.MissingSchemaRelationshipsOutput

    @staticmethod
    def get_unanswerable_question(db_id: str, output: BaseModel) -> list["QuestionUnanswerable"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, MissingSchemaRelationshipsCategory.MissingSchemaRelationshipsOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=MissingSchemaRelationshipsCategory(),
            question=output.question,
            evidence=None,
            sql=None,
            hidden_knowledge=None,
            is_solvable=MissingSchemaRelationshipsCategory.is_solvable()
        )]