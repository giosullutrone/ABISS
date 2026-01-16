from dataclasses import dataclass, asdict
from dataset_dataclasses.question import Question, QuestionUnanswerable
from dataset_dataclasses.system import SystemResponseQuestion, SystemResponseSQL
from prompts import RelevancyLabel, UserKnowledgeLevel
from enum import Enum


@dataclass
class Interaction:
    """
    Dataclass representing an interaction between the system and the user.

    @param system_response: The system's response, either a question or SQL
    @param user_response: The user's response (only for unanswerable questions and if system_response is a question)
    """
    system_response: SystemResponseQuestion | SystemResponseSQL
    user_response: str | None = None

    def to_dict(self) -> dict:
        return asdict(self, dict_factory=lambda x: {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in x})

    @classmethod
    def from_dict(cls, d: dict) -> "Interaction":
        system_response: dict = d.pop("system_response")
        sr = SystemResponseQuestion(**system_response) if system_response.get("_type") == "SystemResponseQuestion" else SystemResponseSQL(**system_response)
        return cls(system_response=sr, user_response=d.get("user_response"))


@dataclass
class InteractionEvaluated(Interaction):
    """
    Dataclass representing an interaction with evaluation metrics.

    @param system_response: The system's response, either a question or SQL
    @param user_response: The user's response (only for unanswerable questions and if system_response is a question)
    @param recognition: Whether the system correctly recognized if the question is answerable or not
    @param classification: Whether the system correctly classified the category (only for unanswerable questions)
    @param relevance: Whether the system's generated question is relevant (only for unanswerable questions)
    @param solved: Whether the final SQL generated is correct i.e. the ambiguity is resolved (only for unanswerable questions and not for the first interaction)
    """
    recognition: bool | None = None
    classification: bool | None = None
    relevance: RelevancyLabel | None = None
    solved: bool | None = None

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update(
            {
                "recognition": self.recognition,
                "classification": self.classification,
                "relevance": self.relevance,
                "solved": self.solved,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict) -> "InteractionEvaluated":
        base_interaction = Interaction.from_dict(d)
        return cls(
            system_response=base_interaction.system_response,
            user_response=base_interaction.user_response,
            recognition=d.get("recognition"),
            classification=d.get("classification"),
            relevance=d.get("relevance"),
            solved=d.get("solved"),
        )


@dataclass
class Conversation:
    question: Question | QuestionUnanswerable
    user_knowledge_level: UserKnowledgeLevel
    interactions: list[InteractionEvaluated]

    def to_dict(self) -> dict:
        return asdict(self, dict_factory=lambda x: {
            k: (v.to_dict() if hasattr(v, "to_dict") else (v.value if isinstance(v, Enum) else v)) 
            for k, v in x
        })

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        question: dict = d.pop("question")
        try:
            q = QuestionUnanswerable.from_dict(question)
        except:
            q = Question(**question)
        interactions = [InteractionEvaluated.from_dict(i) for i in d.pop("interactions")]
        return cls(question=q, user_knowledge_level=UserKnowledgeLevel(d.get("user_knowledge_level")), interactions=interactions)


@dataclass
class Results:
    agent_name: str
    user_models: list[str]
    dataset_name: str
    conversations: list[Conversation]

    def to_dict(self) -> dict:
        return asdict(self, dict_factory=lambda x: {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in x})

    @classmethod
    def from_dict(cls, d: dict) -> "Results":
        conversations = [Conversation.from_dict(c) for c in d.pop("conversations")]
        return cls(
            agent_name=d["agent_name"],
            user_models=d["user_models"],
            dataset_name=d["dataset_name"],
            conversations=conversations,
        )
