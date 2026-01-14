from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import QuestionUnanswerable, Question


class MissingUserKnowledgeCategory(Category):
    class MissingUserKnowledgeOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question that contains user-specific references (e.g., 'my department', 'our projects', 'my courses') whose interpretation depends on objective but user-specific facts not present in the database. The question is valid but cannot be answered without knowing context about the specific user asking the question.")]
        sql_with_user_knowledge: Annotated[str, Field(description="The SQL query that would correctly answer the question if the user-specific knowledge were known (e.g., using a concrete department value instead of the unresolved 'my department' reference).")]
        hidden_knowledge: Annotated[str, Field(description="The hidden user-specific fact that would resolve the ambiguity (e.g., 'The user's department is Engineering' or 'The user is employee ID 12345').")]

    @staticmethod
    def get_name() -> str:
        return "Missing User Knowledge"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return "A question is unanswerable due to Missing User Knowledge when its interpretation depends on objective but user-specific facts absent from the database or knowledge base. These questions contain references to user context (e.g., 'my', 'our', 'I') that require knowing who is asking the question and their specific attributes or affiliations. Without this user-specific information, the question cannot be properly resolved into a concrete SQL query."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the students in my department.",  # Requires knowing user's department
            "Show all projects I'm working on.",  # Requires knowing user's identity
            "What courses am I enrolled in?",  # Requires knowing user's student ID
            "Find employees in my office location.",  # Requires knowing user's office
            "Display our team's sales figures."  # Requires knowing user's team
        ]

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return MissingUserKnowledgeCategory.MissingUserKnowledgeOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel) -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, MissingUserKnowledgeCategory.MissingUserKnowledgeOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=MissingUserKnowledgeCategory(),
            question=output.question,
            evidence=None,
            sql=output.sql_with_user_knowledge,
            hidden_knowledge=output.hidden_knowledge,
            is_solvable=MissingUserKnowledgeCategory.is_solvable()
        )]
