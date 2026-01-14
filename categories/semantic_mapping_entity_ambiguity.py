from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import QuestionUnanswerable, Question


class SemanticMappingEntityAmbiguityCategory(Category):
    class SemanticMappingEntityAmbiguityOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question where a term or expression can correspond to attributes from multiple different entities or tables in the schema. The ambiguity arises because the same concept could be represented in different tables representing different entities or contexts (e.g., enrollment_date in students table vs. student_courses table).")]
        sql_first_entity: Annotated[str, Field(description="The SQL query using the attribute from the first entity (e.g., using students.enrollment_date to represent when the student enrolled at the university).")]
        sql_second_entity: Annotated[str, Field(description="The SQL query using the attribute from the second entity (e.g., using student_courses.enrollment_date to represent when the student enrolled in a specific course).")]
        hidden_knowledge_first_entity: Annotated[str, Field(description="The hidden user intent clarifying that the term refers to the attribute from the first entity (e.g., 'I mean the date when students first enrolled at the university, not in specific courses').")]
        hidden_knowledge_second_entity: Annotated[str, Field(description="The hidden user intent clarifying that the term refers to the attribute from the second entity (e.g., 'I mean the date when students enrolled in the specific course, not their university enrollment date').")]

    @staticmethod
    def get_name() -> str:
        return "Semantic Mapping Ambiguity"

    @staticmethod
    def get_subname() -> str | None:
        return "Entity Ambiguity"

    @staticmethod
    def get_definition() -> str:
        return "Entity Ambiguity occurs when a term or expression in the question can correspond to attributes from multiple different entities or tables in the schema. The same concept mentioned in natural language exists in different tables representing different entities or contexts (e.g., enrollment_date in both students and student_courses tables). The different entity interpretations typically require accessing different tables or following different join paths, as they refer to the same concept but in different relational contexts."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the enrollment date of the students of the 'database' course.",  # students.enrollment_date vs student_courses.enrollment_date
            "Show the start date for all projects in the engineering department.",  # projects.start_date vs departments.start_date vs employees.start_date
            "Find the location of employees working on the Mars project.",  # employees.location vs projects.location vs offices.location
            "What is the price for items in the electronics category?"  # items.price vs categories.base_price
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
    def get_question(db_id: str, output: BaseModel) -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, SemanticMappingEntityAmbiguityCategory.SemanticMappingEntityAmbiguityOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=SemanticMappingEntityAmbiguityCategory(),
            question=output.question,
            evidence=None,
            sql=sql,
            hidden_knowledge=hk,
            is_solvable=SemanticMappingEntityAmbiguityCategory.is_solvable()
        ) for sql, hk in [
            (output.sql_first_entity, output.hidden_knowledge_first_entity),
            (output.sql_second_entity, output.hidden_knowledge_second_entity)
        ]]

