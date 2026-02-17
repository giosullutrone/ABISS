from pydantic import Field, BaseModel
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class ImproperQuestionCategory(Category):
    class ImproperQuestionOutput(BaseModel):
        question: Annotated[str, Field(description="A question that is unrelated to the database domain and cannot be answered through SQL queries. This includes chit-chat, philosophical questions, requests for external reasoning, or commands that are not database queries (e.g., updates, general knowledge questions).")]
        feedback: Annotated[str, Field(description="Explanation of why this question is not solvable, specifying why it is improper (e.g., chit-chat, off-topic, non-query request, general knowledge, etc.) and unrelated to the database domain.")]

    @staticmethod
    def get_name() -> str:
        return "Improper Question"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return "A question is Improper when it is unrelated to the domain of either the database or the knowledge base. This includes casual conversation or greetings (chit-chat), questions requiring external reasoning or general knowledge not grounded in the database domain, requests to perform actions rather than retrieve information (such as updates or modifications), or any other input that is fundamentally not a database query request."

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "Hello! (Improper: chit-chat greeting, not a database query)",
            "How are you doing today? (Improper: chit-chat, not a database query)",
            "What is the meaning of life? (Improper: philosophical question requiring external reasoning, unrelated to the database)",
            "Update my mailing address. (Improper: a data modification request, not a retrieval query)",
            "Who won the World Cup last year? (Improper: general knowledge question unrelated to the database domain)",
            "Can you help me write a poem? (Improper: creative writing request, not a database query)"
        ]

    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return False

    @staticmethod
    def get_output() -> type[BaseModel]:
        return ImproperQuestionCategory.ImproperQuestionOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, ImproperQuestionCategory.ImproperQuestionOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=ImproperQuestionCategory(),
            question=output.question,
            evidence=None,
            sql=None,
            hidden_knowledge=output.feedback,
            is_solvable=ImproperQuestionCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        )]
