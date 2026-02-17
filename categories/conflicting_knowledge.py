from pydantic import Field, BaseModel
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class ConflictingKnowledgeCategory(Category):
    class ConflictingKnowledgeOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question that references a concept for which multiple, non-equivalent definitions or interpretations exist in the knowledge base. The questions itself does not mention the ambiguity or the conflicting evidence, but is answerable by following either interpretation.")]
        evidence_first: Annotated[str, Field(description="The first piece of evidence from the knowledge base that defines or interprets the concept (e.g., 'A student's performance is their average grade').")]
        evidence_second: Annotated[str, Field(description="The second, conflicting piece of evidence from the knowledge base (e.g., 'A student's performance is the average grade weighted by course credits').")]
        hidden_knowledge_first_evidence: Annotated[str, Field(description="The hidden user intent clarifying that the first evidence interpretation should be used (e.g., 'Use the simple average grade definition of performance').")]
        hidden_knowledge_second_evidence: Annotated[str, Field(description="The hidden user intent clarifying that the second evidence interpretation should be used (e.g., 'Use the credit-weighted average definition of performance').")]
        sql_first_evidence: Annotated[str, Field(description="The SQL query based on the first piece of evidence from the knowledge base (e.g., calculating performance as simple average grade).")]
        sql_second_evidence: Annotated[str, Field(description="The SQL query based on the second piece of evidence from the knowledge base (e.g., calculating performance as credit-weighted average grade).")]

    @staticmethod
    def get_name() -> str:
        return "Conflicting Knowledge"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return "A question is ambiguous due to Conflicting Knowledge when a hypothetical retrieval system returns multiple, mutually exclusive policies or evidence definitions for the same concept. The ambiguity does NOT arise from the question wording itself being vague, but from having multiple documented, conflicting interpretations retrieved from the knowledge base. Each piece of evidence provides a valid but non-equivalent definition, leading to structurally different SQL queries. The user must specify which policy/evidence to follow."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the top five students' performance. (Conflicting evidence: one source defines performance as the simple average grade, another as the credit-weighted average grade)",
            "Show the most profitable products. (Conflicting evidence: one source defines profit as revenue minus cost, another as the profit margin percentage)",
            "What is the total compensation for each employee? (Conflicting evidence: one source defines compensation as salary only, another as salary plus benefits)",
            "Rank the departments by efficiency. (Conflicting evidence: one source defines efficiency as output divided by input, another as cost per unit produced)"
        ]
    
    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return ConflictingKnowledgeCategory.ConflictingKnowledgeOutput
    
    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, ConflictingKnowledgeCategory.ConflictingKnowledgeOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=ConflictingKnowledgeCategory(),
            question=output.question,
            evidence="Evidence 1: " + output.evidence_first + "\nEvidence 2: " + output.evidence_second,
            sql=sql,
            hidden_knowledge=hk,
            is_solvable=ConflictingKnowledgeCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        ) for sql, hk in [
            (output.sql_first_evidence, output.hidden_knowledge_first_evidence),
            (output.sql_second_evidence, output.hidden_knowledge_second_evidence)
        ]]
