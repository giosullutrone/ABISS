from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import QuestionUnanswerable


class StructureAmbiguityAttachmentCategory(Category):
    class StructureAmbiguityAttachmentOutput(BaseModel):
        reasoning: Annotated[str, Field(description="Use this field to think step-by-step about how to construct the question and SQL queries. First, identify which modifier or condition will create the ambiguity. Second, determine which elements it could attach to (last only vs. all elements in the conjunction). Third, explain how the SQL filtering will differ between these interpretations. Fourth, confirm why both attachment interpretations are linguistically plausible. Use this reasoning to guide the generation of the 'question', 'sql_last_only', 'sql_all_elements', 'hidden_knowledge_last_only', and 'hidden_knowledge_all_elements' fields.")]
        question: Annotated[str, Field(description="The generated natural language question containing attachment ambiguity, where a modifier or condition can attach to either only the nearest element or to multiple elements in a conjunction or list, leading to different scopes of filtering or constraint.")]
        sql_last_only: Annotated[str, Field(description="The SQL query where the modifier applies only to the last element in the conjunction (e.g., in 'professors and students in engineering', the condition 'in engineering' filters only students, not professors).")]
        sql_all_elements: Annotated[str, Field(description="The SQL query where the modifier applies to all elements in the conjunction (e.g., in 'professors and students in engineering', the condition 'in engineering' filters both professors and students).")]
        hidden_knowledge_last_only: Annotated[str, Field(description="The hidden user intent specifying that the modifier or condition applies only to the last element in the conjunction or list (e.g., 'I want all professors and only the students who are in engineering').")]
        hidden_knowledge_all_elements: Annotated[str, Field(description="The hidden user intent specifying that the modifier or condition applies to all elements in the conjunction or list (e.g., 'I want both professors and students, but only those who are in engineering').")]

    @staticmethod
    def get_name() -> str:
        return "Structure Ambiguity"

    @staticmethod
    def get_subname() -> str | None:
        return "Attachment Ambiguity"

    @staticmethod
    def get_definition() -> str:
        return "Attachment Ambiguity occurs from uncertainty in how a modifier attaches within a sentence containing conjunctions or lists. The modifier (typically a prepositional phrase, clause, or condition) may attach to only the nearest element or to multiple elements in the conjunction, leading to distinct filtering conditions in the resulting query. For example, in 'List the professors and students in engineering', the phrase 'in engineering' may modify only 'students' (narrow attachment) or the entire conjunction 'professors and students' (wide attachment)."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the professors and students in engineering.",
            "Show me the cars and motorcycles manufactured in Japan.",
            "Find the books and articles about machine learning.",
            "Display the managers and employees in the sales department."
        ]

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return StructureAmbiguityAttachmentCategory.StructureAmbiguityAttachmentOutput

    @staticmethod
    def get_unanswerable_question(db_id: str, output: BaseModel) -> list["QuestionUnanswerable"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, StructureAmbiguityAttachmentCategory.StructureAmbiguityAttachmentOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=StructureAmbiguityAttachmentCategory(),
            question=output.question,
            evidence=None,
            sql=sql,
            hidden_knowledge=hk,
            is_solvable=StructureAmbiguityAttachmentCategory.is_solvable()
        ) for sql, hk in [
            (output.sql_last_only, output.hidden_knowledge_last_only),
            (output.sql_all_elements, output.hidden_knowledge_all_elements)
        ]]