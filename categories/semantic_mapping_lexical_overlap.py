from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class SemanticMappingLexicalOverlapCategory(Category):
    class SemanticMappingLexicalOverlapOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question where a term partially matches multiple columns or values with similar names within the SAME entity or closely related entities (e.g., 'email' mapping to personal_email vs. institutional_email in the same students table, or 'heart' matching 'heart failure' vs. 'heart attack' in the same conditions table). The ambiguity is about WHICH COLUMN VARIANT the term maps to, NOT about cross-table/cross-entity mapping, NOT about sentence structure, NOT about user identity, and NOT about vague terminology.")]
        hidden_knowledge_first_mapping: Annotated[str, Field(description="A statement clarifying that the term maps to the first column variant. It should identify which specific column or value the term maps to and what it represents. For example: 'The term email refers to the column personal_email, meaning the students personal email addresses.'")]
        hidden_knowledge_second_mapping: Annotated[str, Field(description="A statement clarifying that the term maps to the second column variant. It should identify which specific column or value the term maps to and what it represents. For example: 'The term email refers to the column institutional_email, meaning the students institutional email addresses.'")]
        sql_first_mapping: Annotated[str, Field(description="A valid, executable SQL query using the first column variant. The two SQL variants must differ only in which column (or value substring) they select or filter on, NOT in which table they query. The query must correctly answer the question under this interpretation. Do not include columns from the other column variant's interpretation in the SELECT clause or filters.")]
        sql_second_mapping: Annotated[str, Field(description="A valid, executable SQL query using the second column variant. The two SQL variants must differ only in which column (or value substring) they select or filter on, NOT in which table they query. The query must correctly answer the question under this interpretation. Do not include columns from the other column variant's interpretation in the SELECT clause or filters.")]

    @staticmethod
    def get_name() -> str:
        return "Semantic Mapping Ambiguity"

    @staticmethod
    def get_subname() -> str | None:
        return "Lexical Overlap"

    @staticmethod
    def get_definition() -> str:
        return (
            "Lexical Overlap Ambiguity arises when a term in the question partially matches multiple distinct columns or values "
            "within the SAME entity or closely related entities, due to shared substrings or naming conventions. "
            "This makes it unclear whether the term refers to a specific attribute variant "
            "(e.g., 'email' mapping to 'personal_email' vs. 'institutional_email' in the same table) "
            "or a specific value substring (e.g., 'heart' matching 'heart failure' vs. 'heart attack' in the same column). "
            "The ambiguity stems from linguistic similarity between column names or data values. "
            "Important: This is NOT about the same concept existing in DIFFERENT tables representing different entities "
            "(e.g., enrollment_date in students vs. student_courses — that is Entity Ambiguity), "
            "NOT about how a modifier attaches in a conjunction (Attachment Ambiguity), "
            "NOT about quantifier scope (Scope Ambiguity), "
            "NOT about user-specific references like 'my' or 'our' (Missing User Knowledge), "
            "NOT about conflicting evidence definitions (Conflicting Knowledge), "
            "and NOT about vague terms with imprecise boundaries like 'recent' or 'high' (Lexical Vagueness)."
        )

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the emails of students in the 'database' course. (Lexical overlap: 'emails' could map to students.personal_email or students.institutional_email — variant columns within the same table)",
            "Show the addresses of customers who ordered laptops. (Lexical overlap: 'addresses' could map to customers.home_address, customers.billing_address, or customers.shipping_address — variant columns within the same table)",
            "What is the phone number for all employees? (Lexical overlap: 'phone number' could map to employees.work_phone or employees.mobile_phone — variant columns within the same table)",
            "Find patients diagnosed with heart conditions. (Lexical overlap: 'heart conditions' could match 'heart failure' or 'heart attack' — substring overlap in data values within the same column)"
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