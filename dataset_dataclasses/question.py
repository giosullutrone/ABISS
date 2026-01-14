from dataclasses import dataclass, asdict
from categories.category import Category
from categories import get_category_by_name


@dataclass
class Question:
    db_id: str
    question: str
    evidence: str | None
    sql: str | None
    category: Category

    def to_dict(self) -> dict:
        return asdict(self, dict_factory=lambda x: {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in x})

@dataclass
class QuestionUnanswerable(Question):
    hidden_knowledge: str | None
    is_solvable: bool
    
    @classmethod
    def from_dict(cls, d: dict) -> "QuestionUnanswerable":
        category_dict = d.pop("category")
        category = get_category_by_name(category_dict["name"], category_dict.get("subname"))

        assert category is not None, f"Unknown category: {category_dict}"

        return cls(
            db_id=d["db_id"],
            question=d["question"],
            evidence=d.get("evidence"),
            sql=d.get("sql"),
            category=category,
            hidden_knowledge=d.get("hidden_knowledge"),
            is_solvable=d["is_solvable"],
        )