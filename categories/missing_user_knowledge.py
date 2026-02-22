from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class MissingUserKnowledgeCategory(Category):
    class MissingUserKnowledgeOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question containing user-specific references (e.g., 'my department', 'our projects', 'my courses', 'I am enrolled in') whose interpretation depends on knowing WHO is asking. The question MUST contain explicit user-referencing pronouns or possessives ('my', 'our', 'I', 'me', 'we'). The ambiguity is about user identity, NOT about sentence structure, NOT about which column a word maps to, NOT about vague terms, and NOT about conflicting definitions.")]
        hidden_knowledge: Annotated[str, Field(description="The objective user-specific fact that resolves the ambiguity. It should state a concrete fact about the user's identity or affiliation. For example: 'The user is in the Engineering department.' or 'The user is employee ID 12345.'")]
        sql_with_user_knowledge: Annotated[str, Field(description="A valid, executable SQL query that correctly answers the question using the concrete user-specific fact from the hidden knowledge (e.g., WHERE department = 'Engineering' instead of the unresolved 'my department'). The query must faithfully represent the user's intent once the disambiguation is applied.")]

    @staticmethod
    def get_name() -> str:
        return "Missing User Knowledge"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return (
            "A question is ambiguous due to Missing User Knowledge when its interpretation depends on objective but user-specific facts "
            "absent from the database or knowledge base. "
            "These questions contain explicit references to user context — pronouns or possessives like 'my', 'our', 'I', 'me', 'we' — "
            "that require knowing who is asking the question and their specific attributes or affiliations. "
            "Without this user-specific information, the question cannot be resolved into a concrete SQL query. "
            "The distinguishing marker is the presence of user-referencing language that ties the query to a specific person's identity. "
            "Important: This is NOT about how a modifier attaches in a conjunction (Attachment Ambiguity), "
            "NOT about quantifier scope (Scope Ambiguity), "
            "NOT about which database table or column a word maps to (Entity Ambiguity / Lexical Overlap), "
            "NOT about conflicting evidence definitions (Conflicting Knowledge), "
            "and NOT about vague terms with imprecise boundaries like 'recent' or 'high' (Lexical Vagueness)."
        )

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the students in my department. (Missing user knowledge: requires knowing which department the user belongs to, to resolve 'my')",
            "Show all projects I'm working on. (Missing user knowledge: requires knowing the user's employee ID to resolve 'I')",
            "Display our team's sales figures. (Missing user knowledge: requires knowing which team the user belongs to, to resolve 'our')",
            "How many tasks are assigned to me this week? (Missing user knowledge: requires knowing the user's identity to resolve 'me')"
        ]

    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return MissingUserKnowledgeCategory.MissingUserKnowledgeOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, MissingUserKnowledgeCategory.MissingUserKnowledgeOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=MissingUserKnowledgeCategory(),
            question=output.question,
            evidence=None,
            sql=output.sql_with_user_knowledge,
            hidden_knowledge=output.hidden_knowledge,
            is_solvable=MissingUserKnowledgeCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        )]
