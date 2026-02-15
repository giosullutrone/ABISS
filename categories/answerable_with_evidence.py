from pydantic import Field, BaseModel
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class AnswerableWithEvidenceCategory(Category):
    class AnswerableWithEvidenceOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question that can be directly answered using the database schema and available data, but requires external evidence to understand or resolve an element in the question (e.g., domain-specific terminology, abbreviations, formulas, or contextual information needed for SQL conversion).")]
        evidence_relevant: Annotated[str, Field(description="External evidence that is necessary to understand or resolve an element in the question. This evidence is essential for correctly converting the question to SQL (e.g., 'GPA is calculated as the weighted average of grades', 'Senior employees are those with >5 years experience', 'CS stands for Computer Science').")]
        evidence_unrelated: Annotated[str | None, Field(description="Optional: Additional external evidence that is completely UNRELATED to answering this specific question. If provided, this evidence MUST be about a different topic, domain, or concept that has no bearing on the question. This tests whether the system can correctly filter out irrelevant information and use only the relevant evidence. Only provide if you want to test the system's ability to distinguish relevant from irrelevant evidence, otherwise set to null.")]
        sql: Annotated[str, Field(description="The SQL query that correctly answers the question based on the database schema and the relevant evidence. This SQL must incorporate the information from evidence_relevant.")]

    @staticmethod
    def get_name() -> str:
        return "Answerable"

    @staticmethod
    def get_subname() -> str | None:
        return "With Evidence"

    @staticmethod
    def get_definition() -> str:
        return "A question is Answerable with Evidence when it can be directly answered using a SQL query against the database, but requires external evidence to understand or resolve specific elements in the question. The question clearly maps to the database schema once the evidence is applied, and the evidence provides necessary context such as domain-specific definitions, abbreviations, formulas, or interpretations that are essential for correct SQL conversion. Unlike Missing External Knowledge questions, these questions CAN be answered once the required evidence is provided."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "What is the GPA of students in Biology 101?",  # Requires evidence defining GPA calculation
            "List all senior employees in the Engineering department.",  # Requires evidence defining 'senior'
            "Show CS courses with high enrollment.",  # Requires evidence that CS = Computer Science and what 'high' means
            "Find products with good profit margins."  # Requires evidence defining what constitutes a 'good' margin
        ]
    
    @staticmethod
    def is_answerable() -> bool:
        return True

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return AnswerableWithEvidenceCategory.AnswerableWithEvidenceOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import Question
        import random
        assert isinstance(output, AnswerableWithEvidenceCategory.AnswerableWithEvidenceOutput)
        
        # Construct evidence string
        if output.evidence_unrelated is not None:
            # Both relevant and unrelated evidence - randomize order
            evidences = [output.evidence_relevant, output.evidence_unrelated]
            random.shuffle(evidences)
            evidence = f"Evidence 1: {evidences[0]}\nEvidence 2: {evidences[1]}"
        else:
            # Only relevant evidence
            evidence = output.evidence_relevant
        
        return [Question(
            db_id=db_id,
            category=AnswerableWithEvidenceCategory(),
            question=output.question,
            evidence=evidence,
            sql=output.sql,
            question_style=question_style,
            question_difficulty=question_difficulty
        )]
