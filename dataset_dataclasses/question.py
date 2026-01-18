from dataclasses import dataclass, asdict
from categories.category import Category
from categories import get_category_by_name_and_subname
from enum import Enum


class QuestionDifficulty(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"

class QuestionStyle(Enum):
    FORMAL = "formal"
    COLLOQUIAL = "colloquial"
    IMPERATIVE = "imperative"
    INTERROGATIVE = "interrogative"
    DESCRIPTIVE = "descriptive"
    CONCISE = "concise"

def get_all_question_styles() -> list[QuestionStyle]:
    return [
        QuestionStyle.FORMAL,
        QuestionStyle.COLLOQUIAL,
        QuestionStyle.IMPERATIVE,
        QuestionStyle.INTERROGATIVE,
        QuestionStyle.DESCRIPTIVE,
        QuestionStyle.CONCISE,
    ]

def get_all_question_difficulties() -> list[QuestionDifficulty]:
    return [
        QuestionDifficulty.SIMPLE,
        QuestionDifficulty.MODERATE,
        QuestionDifficulty.COMPLEX,
        QuestionDifficulty.HIGHLY_COMPLEX,
    ]

@dataclass
class Question:
    db_id: str
    question: str
    evidence: str | None
    sql: str | None
    category: Category
    question_style: QuestionStyle
    question_difficulty: QuestionDifficulty

    def to_dict(self) -> dict:
        return asdict(self, dict_factory=lambda x: {
            k: (v.to_dict() if hasattr(v, "to_dict") else (v.value if isinstance(v, Enum) else v)) 
            for k, v in x
        })

@dataclass
class QuestionUnanswerable(Question):
    hidden_knowledge: str | None
    is_solvable: bool
    
    @classmethod
    def from_dict(cls, d: dict) -> "QuestionUnanswerable":
        category_dict = d.pop("category")
        category = get_category_by_name_and_subname(category_dict["name"], category_dict.get("subname"))

        assert category is not None, f"Unknown category: {category_dict}"

        return cls(
            db_id=d["db_id"],
            question=d["question"],
            evidence=d.get("evidence"),
            sql=d.get("sql"),
            category=category,
            question_style=QuestionStyle(d["question_style"]),
            question_difficulty=QuestionDifficulty(d["question_difficulty"]),
            hidden_knowledge=d.get("hidden_knowledge"),
            is_solvable=d["is_solvable"],
        )