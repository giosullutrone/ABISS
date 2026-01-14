from dataclasses import dataclass, asdict
from categories.category import Category


@dataclass
class SystemResponseQuestion:
    question: str
    category: Category | None

    def to_dict(self) -> dict:
        return asdict(self, dict_factory=lambda x: {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in x})


@dataclass
class SystemResponseSQL:
    sql: str

    def to_dict(self) -> dict:
        return asdict(self)