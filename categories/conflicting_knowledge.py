from pydantic import Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import QuestionUnanswerable


class ConflictingKnowledgeCategory(Category):
    class ConflictingKnowledgeOutput(Category.Output):
        reasoning: Annotated[str, Field(description="Use this field to think step-by-step about how to construct the question, evidence pieces, and SQL queries. First, identify what concept in the question has multiple definitions. Second, determine what the conflicting pieces of evidence are and how they differ. Third, explain how each interpretation leads to different SQL queries. Fourth, confirm why both interpretations are valid based on the available evidence. Use this reasoning to guide the generation of the 'question', 'sql_first_evidence', 'sql_second_evidence', 'evidence_first', 'evidence_second', 'hidden_knowledge_first_evidence', and 'hidden_knowledge_second_evidence' fields.")]
        question: Annotated[str, Field(description="A natural language question that references a concept for which multiple, non-equivalent definitions or interpretations exist in the knowledge base. The ambiguity arises not from the question itself but from conflicting evidence about how to interpret a specific term or calculation.")]
        sql_first_evidence: Annotated[str, Field(description="The SQL query based on the first piece of evidence from the knowledge base (e.g., calculating performance as simple average grade).")]
        sql_second_evidence: Annotated[str, Field(description="The SQL query based on the second piece of evidence from the knowledge base (e.g., calculating performance as credit-weighted average grade).")]
        evidence_first: Annotated[str, Field(description="The first piece of evidence from the knowledge base that defines or interprets the concept (e.g., 'A student's performance is their average grade').")]
        evidence_second: Annotated[str, Field(description="The second, conflicting piece of evidence from the knowledge base (e.g., 'A student's performance is the average grade weighted by course credits').")]
        hidden_knowledge_first_evidence: Annotated[str, Field(description="The hidden user intent clarifying that the first evidence interpretation should be used (e.g., 'Use the simple average grade definition of performance').")]
        hidden_knowledge_second_evidence: Annotated[str, Field(description="The hidden user intent clarifying that the second evidence interpretation should be used (e.g., 'Use the credit-weighted average definition of performance').")]

    @staticmethod
    def get_name() -> str:
        return "Conflicting Knowledge"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return "A question is ambiguous due to Conflicting Knowledge if the knowledge base contains multiple, non-equivalent pieces of evidence for interpreting the same concept in the question. The ambiguity arises not from the natural language question itself, but from inconsistency within the knowledge base about how certain terms, calculations, or concepts should be defined. Different pieces of evidence lead to structurally different SQL queries, and without clarification about which evidence to use, the question cannot be uniquely resolved."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the top five students' performance.",  # Performance = average grade vs. weighted average
            "Show the most profitable products.",  # Profit = revenue - cost vs. profit = margin percentage
            "Find experienced employees.",  # Experience = years at company vs. years in industry
            "Display high-priority tasks."  # Priority based on deadline vs. priority based on importance score
        ]

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[Category.Output]:
        return ConflictingKnowledgeCategory.ConflictingKnowledgeOutput

    @staticmethod
    def get_unanswerable_question(db_id: str, output: Category.Output) -> list["QuestionUnanswerable"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, ConflictingKnowledgeCategory.ConflictingKnowledgeOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=ConflictingKnowledgeCategory(),
            question=output.question,
            evidence="1. " + output.evidence_first + "\n2. " + output.evidence_second,
            sql=sql,
            hidden_knowledge=hk,
            is_solvable=ConflictingKnowledgeCategory.is_solvable()
        ) for sql, hk in [
            (output.sql_first_evidence, output.hidden_knowledge_first_evidence),
            (output.sql_second_evidence, output.hidden_knowledge_second_evidence)
        ]]
