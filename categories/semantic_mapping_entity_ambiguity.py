from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class SemanticMappingEntityAmbiguityCategory(Category):
    class SemanticMappingEntityAmbiguityOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question where a term maps to semantically similar attributes in DIFFERENT tables representing distinct real-world entities (e.g., 'enrollment_date' in a students table vs. a student_courses table). The ambiguity is about WHICH ENTITY the term refers to, NOT about similar column names within the same table, NOT about sentence structure, NOT about user identity, and NOT about vague terminology.")]
        hidden_knowledge_first_entity: Annotated[str, Field(description="A statement clarifying that the term refers to the attribute from the first entity. It should identify the specific table and column and explain the entity-specific meaning. For example: 'The term enrollment date refers to students.enrollment_date, meaning the date the student enrolled at the university.'")]
        hidden_knowledge_second_entity: Annotated[str, Field(description="A statement clarifying that the term refers to the attribute from the second entity. It should identify the specific table and column and explain the entity-specific meaning. For example: 'The term enrollment date refers to student_courses.enrollment_date, meaning the date the student enrolled in a specific course.'")]

    @staticmethod
    def get_name() -> str:
        return "Semantic Mapping Ambiguity"

    @staticmethod
    def get_subname() -> str | None:
        return "Entity Ambiguity"

    @staticmethod
    def get_definition() -> str:
        return (
            "Entity Ambiguity occurs when a term or expression in the question corresponds to semantically similar attributes "
            "belonging to multiple DISTINCT entities or tables in the schema. "
            "The ambiguity arises because the same concept (e.g., 'date', 'location', 'price') exists in different tables "
            "representing separate real-world objects or contexts "
            "(e.g., 'enrollment date' in the students table for university admission vs. in the student_courses table for course registration). "
            "Resolving this requires identifying the correct table or relational path that aligns with the user's intent. "
            "Important: This is NOT about similar/variant column names within the SAME entity "
            "(e.g., personal_email vs. institutional_email in a single students table — that is Lexical Overlap), "
            "NOT about how a modifier attaches in a conjunction (Attachment Ambiguity), "
            "NOT about quantifier scope (Scope Ambiguity), "
            "NOT about user-specific references like 'my' or 'our' (Missing User Knowledge), "
            "NOT about conflicting evidence definitions (Conflicting Knowledge), "
            "and NOT about vague terms with imprecise boundaries like 'recent' or 'high' (Lexical Vagueness)."
        )

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the enrollment date of students in the 'database' course. (Entity ambiguity: 'enrollment date' exists in both the students table and the student_courses table — same concept representing different events in different entity tables)",
            "Show the rating for restaurants in downtown. (Entity ambiguity: 'rating' could refer to restaurants.avg_rating or reviews.rating — same concept in different entity tables with different meanings)",
            "Find the location of employees working on the Mars project. (Entity ambiguity: 'location' could refer to employees.location, projects.location, or offices.location — same concept across different entity tables)",
            "What is the contact number for suppliers of electronic parts? (Entity ambiguity: 'contact number' could refer to suppliers.phone or supplier_contacts.phone — same concept in different entity tables)"
        ]

    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return SemanticMappingEntityAmbiguityCategory.SemanticMappingEntityAmbiguityOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, SemanticMappingEntityAmbiguityCategory.SemanticMappingEntityAmbiguityOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=SemanticMappingEntityAmbiguityCategory(),
            question=output.question,
            evidence=None,
            sql=None,
            hidden_knowledge=hk,
            is_solvable=SemanticMappingEntityAmbiguityCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        ) for hk in [
            output.hidden_knowledge_first_entity,
            output.hidden_knowledge_second_entity
        ]]

