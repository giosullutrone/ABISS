from pydantic import Field, BaseModel
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class ConflictingKnowledgeCategory(Category):
    class ConflictingKnowledgeOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question that references a concept for which multiple, non-equivalent definitions exist in the knowledge base. The question itself is clear and unambiguous in its wording — the ambiguity comes entirely from the knowledge base providing conflicting evidence, NOT from sentence structure, NOT from which column a word maps to, NOT from user identity, and NOT from vague terminology.")]
        evidence_first: Annotated[str, Field(description="The first evidence definition from the knowledge base. It should clearly state the concept and how it is defined under this interpretation. For example: 'A student performance is defined as the simple average grade.'")]
        evidence_second: Annotated[str, Field(description="The second, conflicting evidence definition from the knowledge base. It must define the SAME concept differently, providing a non-equivalent alternative. For example: 'A student performance is defined as the credit-weighted average grade.'")]
        hidden_knowledge_first_evidence: Annotated[str, Field(description="A statement clarifying which evidence definition the user intends. It should unambiguously point to the first definition. For example: 'Use the simple average grade definition of performance.'")]
        hidden_knowledge_second_evidence: Annotated[str, Field(description="A statement clarifying which evidence definition the user intends. It should unambiguously point to the second definition. For example: 'Use the credit-weighted average definition of performance.'")]

    @staticmethod
    def get_name() -> str:
        return "Conflicting Knowledge"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return (
            "A question is ambiguous due to Conflicting Knowledge when a hypothetical retrieval system returns multiple, "
            "mutually exclusive policies or evidence definitions for the same concept. "
            "The ambiguity does NOT arise from the question wording itself being vague or structurally ambiguous, "
            "but entirely from having multiple documented, conflicting interpretations retrieved from the knowledge base. "
            "Each piece of evidence provides a valid but non-equivalent definition, "
            "leading to structurally different SQL queries with different computation logic. "
            "The user must specify which policy or evidence to follow. "
            "Important: This is NOT about how a modifier attaches in a conjunction (Attachment Ambiguity), "
            "NOT about quantifier scope (Scope Ambiguity), "
            "NOT about which database table or column a word maps to (Entity Ambiguity / Lexical Overlap), "
            "NOT about user-specific references like 'my' or 'our' (Missing User Knowledge), "
            "and NOT about vague terms with imprecise boundaries like 'recent' or 'high' (Lexical Vagueness)."
        )

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the top five students by performance. (Conflicting evidence: one source defines performance as the simple average grade, another as the credit-weighted average grade)",
            "Show the most profitable products. (Conflicting evidence: one source defines profit as revenue minus cost, another as the profit margin percentage)",
            "Which employees are eligible for a bonus? (Conflicting evidence: one policy defines eligibility as working over 40 hours per week, another as achieving a performance rating above 4.0)",
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
            sql=None,
            hidden_knowledge=hk,
            is_solvable=ConflictingKnowledgeCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        ) for hk in [
            output.hidden_knowledge_first_evidence,
            output.hidden_knowledge_second_evidence
        ]]
