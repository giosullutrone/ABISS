from pydantic import Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import QuestionUnanswerable


class StructureAmbiguityScopeCategory(Category):
    class StructureAmbiguityScopeOutput(Category.Output):
        reasoning: Annotated[str, Field(description="Use this field to think step-by-step about how to construct the question and SQL queries. First, identify which quantifier (e.g., 'each', 'every', 'all') will create the ambiguity. Second, explain how it can be interpreted collectively (all entities together) vs. distributively (each entity independently). Third, determine what the structural difference will be in the SQL queries (aggregation vs. grouping). Fourth, confirm why both interpretations are reasonable given the question's wording. Use this reasoning to guide the generation of the 'question', 'sql_collective', 'sql_distributive', 'hidden_knowledge_collective', and 'hidden_knowledge_distributive' fields.")]
        question: Annotated[str, Field(description="The generated natural language question containing scope ambiguity, where quantifiers (e.g., 'each', 'every', 'all') can be interpreted either collectively (referring to all entities together) or distributively (treating each entity independently).")]
        sql_collective: Annotated[str, Field(description="The SQL query for the collective interpretation, where the quantifier refers to all entities together, typically using aggregation over the entire group (e.g., 'What courses does each department offer?' interpreted as all courses offered across all departments).")]
        sql_distributive: Annotated[str, Field(description="The SQL query for the distributive interpretation, where the quantifier treats each entity independently, typically using grouping or iteration (e.g., 'What courses does each department offer?' interpreted as courses grouped by individual departments).")]
        hidden_knowledge_collective: Annotated[str, Field(description="The hidden user intent specifying that the quantifier should be interpreted collectively, referring to all entities as a group rather than individually (e.g., 'I want all courses offered by any department' rather than 'courses per department').")]
        hidden_knowledge_distributive: Annotated[str, Field(description="The hidden user intent specifying that the quantifier should be interpreted distributively, treating each entity separately and typically requiring grouping in the SQL query (e.g., 'I want to see which courses each individual department offers').")]

    @staticmethod
    def get_name() -> str:
        return "Structure Ambiguity"

    @staticmethod
    def get_subname() -> str | None:
        return "Scope Ambiguity"

    @staticmethod
    def get_definition() -> str:
        return "Scope Ambiguity arises from unclear quantifiers (e.g., 'each', 'every', 'all') in a question. The ambiguity stems from whether these quantifiers should be interpreted collectively, referring to all entities together, or distributively, treating each entity independently. These interpretations yield structurally different SQL queries: collective interpretations typically use simple aggregation over the entire dataset, while distributive interpretations require grouping or iteration to handle each entity separately."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "What courses does each department offer?",
            "List all projects every employee worked on.",
            "Show the total sales for each region.",
            "What are all the books each author has written?"
        ]

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[Category.Output]:
        return StructureAmbiguityScopeCategory.StructureAmbiguityScopeOutput

    @staticmethod
    def get_unanswerable_question(db_id: str, output: Category.Output) -> list["QuestionUnanswerable"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, StructureAmbiguityScopeCategory.StructureAmbiguityScopeOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=StructureAmbiguityScopeCategory(),
            question=output.question,
            evidence=None,
            sql=sql,
            hidden_knowledge=hk,
            is_solvable=StructureAmbiguityScopeCategory.is_solvable()
        ) for sql, hk in [
            (output.sql_collective, output.hidden_knowledge_collective),
            (output.sql_distributive, output.hidden_knowledge_distributive)
        ]]