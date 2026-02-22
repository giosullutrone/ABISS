"""Unified user response: produces both relevancy label and answer in a single
model call, then selects the best answer among label-consistent candidates.

This replaces the separate QuestionRelevancy + UserAnswer two-step pipeline,
eliminating ground-truth SQL leakage by never exposing the full GT SQL to any
user-simulator prompt.

Answer selection only considers candidates from models that agreed with the
majority-voted relevancy label, avoiding evaluation of answers generated under
a different classification intent.
"""

from typing import Callable, cast
from db_datasets.db_dataset import DBDataset
from models.model import Model
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.question import QuestionUnanswerable
from users.best_user_answer import BestUserAnswer
from users.prompts.user_response_prompt import (
    get_user_response_prompt_solvable,
    UserResponseSolvableModel,
    get_user_response_solvable_result,
    get_user_response_prompt_answerable,
    UserResponseAnswerableModel,
    get_user_response_answerable_result,
)
from pydantic import BaseModel


class UserResponse:
    """Single-step user response: classifies relevancy and generates an answer."""

    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db = db
        self.models = models
        self.best_user_answer = BestUserAnswer(db, models)

    def get_response(self, conversations: list[Conversation]) -> None:
        """Classify relevancy AND generate answer for each conversation in one step.

        Unsolvable questions are short-circuited (IRRELEVANT + canned refusal).
        All other conversations go through the unified prompt.
        """
        # ---- short-circuit unsolvable questions ----
        to_classify: list[int] = []
        prompts: list[str] = []
        model_classes: list[type[BaseModel]] = []
        result_extractors: list[Callable] = []

        for i, conv in enumerate(conversations):
            is_answerable = conv.question.category.is_answerable()
            is_solvable = (
                conv.question.category.is_solvable()
                if isinstance(conv.question, QuestionUnanswerable)
                else True
            )

            if not is_solvable:
                # Unsolvable: no interaction can resolve it.
                conv.interactions[-1].relevance = RelevancyLabel.IRRELEVANT
                conv.interactions[-1].user_response = (
                    "That's not relevant to my question."
                )
                continue

            to_classify.append(i)
            if is_answerable:
                prompts.append(get_user_response_prompt_answerable(conv))
                model_classes.append(UserResponseAnswerableModel)
                result_extractors.append(get_user_response_answerable_result)
            else:
                prompts.append(get_user_response_prompt_solvable(conv))
                model_classes.append(UserResponseSolvableModel)
                result_extractors.append(get_user_response_solvable_result)

        if not prompts:
            return  # all unsolvable

        # ---- generate (relevancy, answer) pairs from every model ----
        # relevancy_votes[prompt_idx] = {label: count}
        relevancy_votes: list[dict[RelevancyLabel, int]] = [
            {RelevancyLabel.RELEVANT: 0, RelevancyLabel.TECHNICAL: 0, RelevancyLabel.IRRELEVANT: 0}
            for _ in prompts
        ]
        # per_model_results[prompt_idx] = list of (label, answer) per model
        per_model_results: list[list[tuple[RelevancyLabel, str]]] = [[] for _ in prompts]

        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(prompts, model_classes)
            model.close()

            for pidx, response in enumerate(responses):
                label, answer = result_extractors[pidx](response)
                relevancy_votes[pidx][label] += 1
                per_model_results[pidx].append((label, answer))

        # ---- majority vote on relevancy ----
        final_labels: list[RelevancyLabel] = []
        for pidx in range(len(prompts)):
            votes = relevancy_votes[pidx]
            sorted_labels = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
            top_label, top_count = sorted_labels[0]
            _, second_count = sorted_labels[1]
            # Ties resolve conservatively: default to IRRELEVANT (no info leaks)
            if top_count == second_count:
                final_labels.append(RelevancyLabel.IRRELEVANT)
            else:
                final_labels.append(top_label)

        # ---- assign relevancy labels ----
        for pidx, conv_idx in enumerate(to_classify):
            conversations[conv_idx].interactions[-1].relevance = final_labels[pidx]

        # ---- filter to label-consistent answers only ----
        consistent_answers: list[list[str]] = []
        for pidx in range(len(prompts)):
            winning_label = final_labels[pidx]
            filtered = [
                answer for label, answer in per_model_results[pidx]
                if label == winning_label
            ]
            # If tie resolved to IRRELEVANT and no model actually voted IRRELEVANT,
            # fall back to all answers (the tournament will pick the best refusal).
            if not filtered:
                filtered = [answer for _, answer in per_model_results[pidx]]
            consistent_answers.append(filtered)

        # ---- select best answer ----
        # If only one label-consistent answer, use it directly (skip tournament).
        best_answers: list[str] = []
        needs_tournament_indices: list[int] = []
        needs_tournament_convs: list[Conversation] = []
        needs_tournament_answers: list[list[str]] = []

        for pidx in range(len(prompts)):
            if len(consistent_answers[pidx]) == 1:
                best_answers.append(consistent_answers[pidx][0])
            else:
                best_answers.append("")  # placeholder
                needs_tournament_indices.append(pidx)
                needs_tournament_convs.append(conversations[to_classify[pidx]])
                needs_tournament_answers.append(consistent_answers[pidx])

        if needs_tournament_convs:
            tournament_results = self.best_user_answer.select_best_user_answers(
                needs_tournament_convs, needs_tournament_answers
            )
            for i, pidx in enumerate(needs_tournament_indices):
                best_answers[pidx] = tournament_results[i]

        for pidx, conv_idx in enumerate(to_classify):
            conversations[conv_idx].interactions[-1].user_response = best_answers[pidx]
