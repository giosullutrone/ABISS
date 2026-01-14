from pydantic import Field
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import QuestionUnanswerable


class ImproperQuestionCategory(Category):
    class ImproperQuestionOutput(Category.Output):
        reasoning: Annotated[str, Field(description="Use this field to think step-by-step about how to construct the question. First, determine what the question should be asking for. Second, explain why it is unrelated to database querying or the domain covered by the database. Third, identify what category of improper question it falls into (chit-chat, external reasoning, non-query request). Fourth, confirm why it cannot be addressed with SQL queries against the database. Use this reasoning to guide the generation of the 'question' field.")]
        question: Annotated[str, Field(description="A question that is unrelated to the database domain and cannot be answered through SQL queries. This includes chit-chat, philosophical questions, requests for external reasoning, or commands that are not database queries (e.g., updates, general knowledge questions).")]

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
            "Hello!",  # Chit-chat
            "How are you doing today?",  # Chit-chat
            "What is the meaning of life?",  # External reasoning/philosophy
            "Update my mailing address.",  # Non-query request (update command)
            "Who won the World Cup last year?",  # General knowledge unrelated to database domain
            "Can you help me write a poem?"  # Request outside database querying
        ]

    @staticmethod
    def is_solvable() -> bool:
        return False

    @staticmethod
    def get_output() -> type[Category.Output]:
        return ImproperQuestionCategory.ImproperQuestionOutput

    @staticmethod
    def get_unanswerable_question(db_id: str, output: Category.Output) -> list["QuestionUnanswerable"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, ImproperQuestionCategory.ImproperQuestionOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=ImproperQuestionCategory(),
            question=output.question,
            evidence=None,
            sql=None,
            hidden_knowledge=None,
            is_solvable=ImproperQuestionCategory.is_solvable()
        )]
