from dataclasses import dataclass
from dataset_dataclasses.question import Question, QuestionUnanswerable
from utils.dataclass_utils import generic_to_dict
from categories.category import Category
from categories import get_category_by_name_and_subname
from enum import Enum


class RelevancyLabel(str, Enum):
    RELEVANT = "Relevant"
    TECHNICAL = "Technical"
    IRRELEVANT = "Irrelevant"

class CategoryUse(Enum):
    GROUND_TRUTH = "ground_truth"
    PREDICTED = "predicted"
    NO_CATEGORY = "no_category"

@dataclass
class SystemResponse:
    system_question: str | None = None
    system_sql: str | None = None
    system_feedback: str | None = None

    def __str__(self) -> str:
        if self.system_question is not None:
            return self.system_question
        if self.system_sql is not None:
            return self.system_sql
        if self.system_feedback is not None:
            return self.system_feedback
        return ""

    def to_dict(self) -> dict:
        return generic_to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SystemResponse":
        return cls(
            system_question=d.get("system_question"),
            system_sql=d.get("system_sql"),
            system_feedback=d.get("system_feedback"),
        )

@dataclass
class Interaction:
    system_response: SystemResponse
    user_response: str | None = None
    relevance: RelevancyLabel | None = None

    def to_dict(self) -> dict:
        return generic_to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Interaction":
        system_response = SystemResponse.from_dict(d.get("system_response", {}))
        return cls(
            system_response=system_response,
            user_response=d.get("user_response"),
            relevance=RelevancyLabel(d.get("relevance")) if d.get("relevance") is not None else None,
        )

@dataclass
class Conversation:
    question: Question
    category_use: CategoryUse
    interactions: list[Interaction]

    predicted_category: Category | None = None
    predicted_sql: str | None = None
    predicted_feedback: str | None = None

    recognition: bool | None = None
    classification: bool | None = None
    solved: bool | None = None
    explained: bool | None = None

    def to_dict(self) -> dict:
        return generic_to_dict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        question = d["question"]
        try:
            q = QuestionUnanswerable.from_dict(question)
        except (KeyError, TypeError, ValueError, AssertionError):
            q = Question.from_dict(question)
        interactions = [Interaction.from_dict(i) for i in d["interactions"]]
        pred_cat_dict = d.get("predicted_category")
        predicted_category = (
            get_category_by_name_and_subname(pred_cat_dict["name"], pred_cat_dict.get("subname"))
            if pred_cat_dict else None
        )
        return cls(
            question=q,
            category_use=CategoryUse(d.get("category_use")),
            interactions=interactions,
            predicted_category=predicted_category,
            predicted_sql=d.get("predicted_sql"),
            predicted_feedback=d.get("predicted_feedback"),
            recognition=d.get("recognition"),
            classification=d.get("classification"),
            explained=d.get("explained"),
            solved=d.get("solved"),
        )

@dataclass
class Results:
    agent_name: str
    user_name: str
    dataset_name: str
    conversations: list[Conversation]

    def to_dict(self) -> dict:
        return generic_to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Results":
        conversations = [Conversation.from_dict(c) for c in d["conversations"]]
        return cls(
            dataset_name=d["dataset_name"],
            user_name=d["user_name"],
            agent_name=d["agent_name"],
            conversations=conversations,
        )
