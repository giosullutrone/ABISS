"""Dataclasses for tracking per-model council votes across the ABISS pipeline.

Used to persist individual model judgments (normally discarded after majority
voting) so that inter-model agreement rates can be analyzed for the paper.
"""

from dataclasses import dataclass, field, asdict
import json
from typing import Any


@dataclass
class ModelVote:
    """A single model's vote on a single question."""
    model_name: str
    vote: bool | int | str  # bool for yes/no, int for pairwise (0/1), str for labels


@dataclass
class QuestionVotes:
    """Per-model votes for a single question at a single validation stage."""
    question_index: int
    question_text: str
    votes: list[ModelVote]
    aggregate_result: bool  # majority decision
    removed: bool  # was the question removed by this stage?


@dataclass
class ValidationStageResult:
    """Result of a single validation stage, including per-model vote tracking.

    `validities` is a drop-in replacement for the old list[bool] return type.
    `question_votes` contains per-model vote details (empty for non-council validators).
    """
    stage_name: str
    validities: list[bool]
    question_votes: list[QuestionVotes] = field(default_factory=list)

    def agreement_rate(self) -> float:
        """Fraction of questions where all models voted the same way.

        Returns 1.0 if there are no questions with votes.
        """
        if not self.question_votes:
            return 1.0
        unanimous = 0
        for qv in self.question_votes:
            if not qv.votes:
                continue
            first = qv.votes[0].vote
            if all(v.vote == first for v in qv.votes):
                unanimous += 1
        return unanimous / len(self.question_votes)


@dataclass
class GenerationTrackingReport:
    """Aggregated council tracking for the entire generation pipeline."""
    stages: list[ValidationStageResult]

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


# ---- Benchmark tracking ----

@dataclass
class RelevancyVotes:
    """Per-model relevancy classifications for a single question at one interaction step."""
    question_index: int
    interaction_step: int
    per_model_labels: list[tuple[str, str]]  # (model_name, label)
    winning_label: str
    unanimous: bool


@dataclass
class TournamentVotes:
    """Pairwise tournament results for selecting the best user answer."""
    question_index: int
    interaction_step: int
    pairwise_results: list[dict[str, Any]]  # {model, candidate_a, candidate_b, winner}
    final_tallies: list[int]
    winning_answer_model: str
    margin: int


@dataclass
class BenchmarkTrackingReport:
    """Aggregated council tracking for the benchmark pipeline."""
    relevancy_votes: list[RelevancyVotes]
    tournament_votes: list[TournamentVotes]
    feedback_votes: list[QuestionVotes]

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
