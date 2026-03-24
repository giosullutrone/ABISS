from pydantic import BaseModel, Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class StructureAmbiguityAttachmentCategory(Category):
    class StructureAmbiguityAttachmentOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question containing attachment ambiguity, where a modifier, condition, or clause syntactically follows a conjunction (e.g., 'A and B [modifier]') and it is unclear whether the modifier attaches only to the nearest element (low attachment) or to the entire conjunction (high attachment). The ambiguity must arise purely from sentence structure, NOT from user-specific references ('my', 'our'), NOT from which table/column a word maps to, and NOT from conflicting definitions.")]
        hidden_knowledge_last_only: Annotated[str, Field(description="A statement clarifying that the modifier attaches only to the nearest element (low attachment). It should clearly identify which element(s) the modifier applies to and which it does not. For example: 'The condition in engineering applies only to students, not to professors.'")]
        hidden_knowledge_all_elements: Annotated[str, Field(description="A statement clarifying that the modifier attaches to the entire conjunction (high attachment). It should clearly state that all elements are affected by the modifier. For example: 'The condition in engineering applies to both professors and students.'")]

    @staticmethod
    def get_name() -> str:
        return "Structure Ambiguity"

    @staticmethod
    def get_subname() -> str | None:
        return "Attachment Ambiguity"

    @staticmethod
    def get_definition() -> str:
        return (
            "Attachment Ambiguity occurs when it is unclear how a modifier, phrase, or clause syntactically attaches to the rest of the sentence. "
            "The ambiguity is purely structural: it arises from the grammatical position of a modifier relative to a conjunction, "
            "not from the meaning of individual words, user identity, or conflicting definitions. "
            "Typically involving prepositional phrases (e.g., 'professors and students in engineering'), "
            "relative clauses (e.g., 'suppliers and manufacturers who are certified'), "
            "or participial phrases (e.g., 'managers and employees hired after 2020'), "
            "the ambiguity has exactly two readings: "
            "'high attachment' where the modifier applies to the entire conjunction (e.g., both professors and students must be in engineering), "
            "versus 'low attachment' where it applies only to the nearest noun (e.g., only students must be in engineering). "
            "These two readings produce structurally different SQL queries with different filtering scopes. "
            "Important: This is NOT about which database table or column a word maps to (Entity Ambiguity / Lexical Overlap), "
            "NOT about user-specific references like 'my' or 'our' (Missing User Knowledge), "
            "NOT about quantifier scope like 'each' or 'every' (Scope Ambiguity), "
            "and NOT about conflicting evidence definitions (Conflicting Knowledge)."
        )

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List the professors and students in engineering. (Attachment ambiguity: does 'in engineering' modify only 'students' or both 'professors and students'?)",
            "Find the suppliers and manufacturers who are certified organic. (Attachment ambiguity: does the relative clause 'who are certified organic' apply only to 'manufacturers' or to both 'suppliers and manufacturers'?)",
            "Display the managers and employees hired after 2020. (Attachment ambiguity: does the participial phrase 'hired after 2020' filter only 'employees' or both 'managers and employees'?)",
            "Show the songs and albums released in 2023. (Attachment ambiguity: does 'released in 2023' apply only to 'albums' or to both 'songs and albums'?)"
        ]

    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return StructureAmbiguityAttachmentCategory.StructureAmbiguityAttachmentOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, StructureAmbiguityAttachmentCategory.StructureAmbiguityAttachmentOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=StructureAmbiguityAttachmentCategory(),
            question=output.question,
            evidence=None,
            sql=None,
            hidden_knowledge=hk,
            is_solvable=StructureAmbiguityAttachmentCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        ) for hk in [
            output.hidden_knowledge_last_only,
            output.hidden_knowledge_all_elements
        ]]