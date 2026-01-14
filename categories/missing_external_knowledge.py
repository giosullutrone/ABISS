from pydantic import Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import QuestionUnanswerable


class MissingExternalKnowledgeCategory(Category):
    class MissingExternalKnowledgeOutput(Category.Output):
        reasoning: Annotated[str, Field(description="Use this field to think step-by-step about how to construct the question. First, determine what the question should be asking for. Second, identify what objective, domain-specific facts or policies are required to answer it. Third, explain why this knowledge cannot be derived from the database schema or data alone. Fourth, consider what external knowledge source would be needed to make the question answerable. Use this reasoning to guide the generation of the 'question' field.")]
        question: Annotated[str, Field(description="A natural language question whose interpretation depends on objective, domain-specific facts, conventions, or policies that are not present in the database. The question appears valid but cannot be answered without external knowledge such as conversion formulas, grading scales, classification rules, or domain-specific standards.")]

    @staticmethod
    def get_name() -> str:
        return "Missing External Knowledge"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return "A question is unanswerable due to Missing External Knowledge when its interpretation depends on objective, domain-specific facts, conventions, or policies not present in the database or knowledge base. This includes cases requiring conversion formulas, classification rules, grading scales, domain standards, or other factual knowledge that exists outside the database but is necessary to correctly answer the question. Without this external knowledge, no valid SQL query can be formulated."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List all the grades of the students for the course 'database' using alphabetic notation (A-F).",  # Requires grade mapping
            "Show the students' GPA on a 4.0 scale.",  # Requires GPA calculation formula
            "Convert all product prices to euros.",  # Requires current exchange rates
            "Classify employees as junior, mid-level, or senior based on their experience.",  # Requires classification rules
            "Find courses that satisfy the computer science degree requirements."  # Requires degree requirement policies
        ]

    @staticmethod
    def is_solvable() -> bool:
        return False

    @staticmethod
    def get_output() -> type[Category.Output]:
        return MissingExternalKnowledgeCategory.MissingExternalKnowledgeOutput

    @staticmethod
    def get_unanswerable_question(db_id: str, output: Category.Output) -> list["QuestionUnanswerable"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, MissingExternalKnowledgeCategory.MissingExternalKnowledgeOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=MissingExternalKnowledgeCategory(),
            question=output.question,
            evidence=None,
            sql=None,
            hidden_knowledge=None,
            is_solvable=MissingExternalKnowledgeCategory.is_solvable()
        )]
