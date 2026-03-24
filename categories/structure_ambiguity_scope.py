from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class StructureAmbiguityScopeCategory(Category):
    class StructureAmbiguityScopeOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question containing scope ambiguity, where a quantifier (e.g., 'each', 'every', 'all', 'any') can be read either collectively (all entities as one group) or distributively (each entity separately). The ambiguity must arise from quantifier scope, NOT from modifier attachment to a conjunction, NOT from which table/column a word maps to, NOT from user-specific references, and NOT from vague terms with imprecise boundaries.")]
        hidden_knowledge_collective: Annotated[str, Field(description="A statement clarifying that the quantifier should be interpreted collectively (all entities as one group). It should specify that the entities are treated as a single combined pool. For example: 'Each department means all departments together — list all courses offered by any department.'")]
        hidden_knowledge_distributive: Annotated[str, Field(description="A statement clarifying that the quantifier should be interpreted distributively (each entity separately). It should specify that results are expected per individual entity. For example: 'Each department means per individual department — list courses grouped by department.'")]

    @staticmethod
    def get_name() -> str:
        return "Structure Ambiguity"

    @staticmethod
    def get_subname() -> str | None:
        return "Scope Ambiguity"

    @staticmethod
    def get_definition() -> str:
        return (
            "Scope Ambiguity arises when it is unclear how broadly a quantifier — such as 'each', 'every', 'all', or 'any' — "
            "ranges over the entities it refers to. The ambiguity is purely about quantifier scope, producing two readings: "
            "a 'collective' interpretation where the quantifier treats all entities as a single group "
            "(e.g., 'each department' meaning the set of all departments combined), "
            "versus a 'distributive' interpretation where each entity is treated separately "
            "(e.g., 'each department' meaning per individual department). "
            "These readings yield structurally different SQL: collective typically uses flat aggregation, "
            "while distributive requires GROUP BY or partitioning. "
            "Important: This is NOT about which noun a modifier attaches to in a conjunction (Attachment Ambiguity), "
            "NOT about which database table or column a word maps to (Entity Ambiguity / Lexical Overlap), "
            "NOT about user-specific references like 'my' or 'our' (Missing User Knowledge), "
            "NOT about conflicting evidence definitions (Conflicting Knowledge), "
            "and NOT about vague terms with imprecise boundaries like 'recent' or 'high' (Lexical Vagueness)."
        )

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "What courses does each department offer? (Scope ambiguity: 'each department' could mean all courses across all departments combined, or courses grouped per individual department)",
            "List all projects every employee worked on. (Scope ambiguity: 'every employee' could mean all projects collectively, or projects listed per individual employee)",
            "Show the total revenue for every store. (Scope ambiguity: 'every store' could mean one grand total across all stores, or a separate total per individual store)",
            "What are the books each author has written? (Scope ambiguity: 'each author' could mean all books collectively, or books grouped by individual author)"
        ]

    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return StructureAmbiguityScopeCategory.StructureAmbiguityScopeOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, StructureAmbiguityScopeCategory.StructureAmbiguityScopeOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=StructureAmbiguityScopeCategory(),
            question=output.question,
            evidence=None,
            sql=None,
            hidden_knowledge=hk,
            is_solvable=StructureAmbiguityScopeCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        ) for hk in [
            output.hidden_knowledge_collective,
            output.hidden_knowledge_distributive
        ]]