from validators.validator import Validator
from categories.category import Category
from dataset_dataclasses.question import QuestionUnanswerable
from tqdm import tqdm


class CheckDuplicate(Validator):
    def __init__(self) -> None:
        pass

    def validate(self, questions: list[QuestionUnanswerable]) -> list[bool]:
        # Check if any questions is a copy of another question in the dataset.
        seen_questions: set[tuple[str, str | None]] = set()
        valids: list[bool] = []
        for question in tqdm(questions, desc="Check Duplicate Validation"):
            valids.append((question.question, question.hidden_knowledge) not in seen_questions)
            seen_questions.add((question.question, question.hidden_knowledge))
        return valids
