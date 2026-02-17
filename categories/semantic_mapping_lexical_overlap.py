from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class SemanticMappingLexicalOverlapCategory(Category):
    class SemanticMappingLexicalOverlapOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question containing a term or expression that can map to multiple schema attributes with similar or identical names within the same entity or closely related entities. The ambiguity arises from lexical overlap where the same expression corresponds to different columns representing variants or types of the same concept (e.g., personal_email vs. institutional_email).")]
        hidden_knowledge_first_mapping: Annotated[str, Field(description="The hidden user intent clarifying that the ambiguous term refers to the first schema mapping (e.g., 'I mean the students' personal email addresses, not their institutional ones').")]
        hidden_knowledge_second_mapping: Annotated[str, Field(description="The hidden user intent clarifying that the ambiguous term refers to the second schema mapping (e.g., 'I mean the students' institutional email addresses, not their personal ones').")]
        sql_first_mapping: Annotated[str, Field(description="The SQL query using the first plausible schema mapping for the ambiguous term (e.g., using students.personal_email to represent the students' personal email addresses).")]
        sql_second_mapping: Annotated[str, Field(description="The SQL query using the second plausible schema mapping for the ambiguous term (e.g., using students.institutional_email to represent the students' institutional email addresses).")]

    @staticmethod
    def get_name() -> str:
        return "Semantic Mapping Ambiguity"

    @staticmethod
    def get_subname() -> str | None:
        return "Lexical Overlap"

    @staticmethod
    def get_definition() -> str:
        return "Lexical Overlap arises when two or more schema attributes share similar or identical forms, making it unclear which specific variant or type of an attribute a term in the question refers to. These attributes typically exist within the same table or closely related tables and represent different variants of the same concept (e.g., personal_email vs. institutional_email, home_address vs. work_address). The same expression in natural language can correspond to multiple columns, each representing a distinct variant, and each interpretation yields a distinct SQL mapping."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the emails of the students of the 'database' course. (Ambiguous: 'emails' could map to students.personal_email or students.institutional_email)",
            "Show the addresses of customers who ordered laptops. (Ambiguous: 'addresses' could map to customers.home_address, customers.billing_address, or customers.shipping_address)",
            "What is the phone number for all employees? (Ambiguous: 'phone number' could map to employees.work_phone or employees.mobile_phone)",
            "Find the start date for projects in the IT department. (Ambiguous: 'start date' could map to projects.planned_start_date or projects.actual_start_date)"
        ]

    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return SemanticMappingLexicalOverlapCategory.SemanticMappingLexicalOverlapOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, SemanticMappingLexicalOverlapCategory.SemanticMappingLexicalOverlapOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=SemanticMappingLexicalOverlapCategory(),
            question=output.question,
            evidence=None,
            sql=sql,
            hidden_knowledge=hk,
            is_solvable=SemanticMappingLexicalOverlapCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        ) for sql, hk in [
            (output.sql_first_mapping, output.hidden_knowledge_first_mapping),
            (output.sql_second_mapping, output.hidden_knowledge_second_mapping)
        ]]