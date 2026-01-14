from validators.validator import Validator
from dataset_dataclasses.question import Question, QuestionUnanswerable
from tqdm import tqdm


class CheckDuplicate(Validator):
    def __init__(self) -> None:
        pass

    def validate(self, questions: list[Question]) -> list[bool]:
        # Check if any questions is a copy of another question in the dataset.
        seen_questions: set[tuple[str, str | None]] = set()
        valids: list[bool] = []
        for question in tqdm(questions, desc="Check Duplicate Validation"):
            valids.append((question.question, question.hidden_knowledge if isinstance(question, QuestionUnanswerable) else None) not in seen_questions)
            seen_questions.add((question.question, question.hidden_knowledge if isinstance(question, QuestionUnanswerable) else None))
        return valids
