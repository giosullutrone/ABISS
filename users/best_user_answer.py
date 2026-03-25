from typing import Callable
from itertools import combinations
import random
from pydantic import BaseModel
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.council_tracking import TournamentVotes
from models.model import Model
from db_datasets.db_dataset import DBDataset
from users.prompts.best_user_answer_prompt import (
    get_best_user_answer_relevant_prompt, BestUserAnswerRelevantResponse, get_best_user_answer_relevant_result,
    get_best_user_answer_technical_prompt, BestUserAnswerTechnicalResponse, get_best_user_answer_technical_result,
    get_best_user_answer_irrelevant_prompt, BestUserAnswerIrrelevantResponse, get_best_user_answer_irrelevant_result
)


class BestUserAnswer:
    """Selects the best user answer among candidates via pairwise tournament
    evaluated by the model council.

    Uses combinations (not permutations) with randomized A/B ordering
    to halve comparison count while mitigating positional bias.
    """

    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def select_best_user_answers(
        self,
        conversations: list[Conversation],
        answers: list[list[str]],
        candidate_model_indices: list[list[int]] | None = None,
        sql_fragments_per_conv: list[list[str]] | None = None,
        conversation_indices: list[int] | None = None,
    ) -> tuple[list[str], list[TournamentVotes]]:
        """Run a pairwise tournament for each conversation's candidate answers.

        Uses combinations (A-vs-B once, not A-vs-B and B-vs-A) with
        randomized candidate ordering to mitigate positional bias.

        Args:
            conversation_indices: Global conversation indices for tracking.
                If None, uses 0-based batch indices.
        """
        votes: list[list[int]] = [[0] * len(ans) for ans in answers]

        # Generate pairwise comparison prompts using combinations
        pairwise_prompts: dict[int, list[tuple[int, int, str, type[BaseModel], Callable]]] = {}
        for i, gens in enumerate(answers):
            pairwise_prompts[i] = []
            assert conversations[i].interactions[-1].relevance is not None
            relevance = conversations[i].interactions[-1].relevance

            for j, k in combinations(range(len(gens)), 2):
                # Randomize A/B ordering to mitigate positional bias
                if random.random() < 0.5:
                    a_idx, b_idx = j, k
                else:
                    a_idx, b_idx = k, j

                if relevance == RelevancyLabel.RELEVANT:
                    prompt = get_best_user_answer_relevant_prompt(
                        self.db, conversations[i], gens[a_idx], gens[b_idx])
                    model_class = BestUserAnswerRelevantResponse
                    extractor = get_best_user_answer_relevant_result
                elif relevance == RelevancyLabel.TECHNICAL:
                    frags = sql_fragments_per_conv[i] if sql_fragments_per_conv else None
                    prompt = get_best_user_answer_technical_prompt(
                        self.db, conversations[i], gens[a_idx], gens[b_idx], frags)
                    model_class = BestUserAnswerTechnicalResponse
                    extractor = get_best_user_answer_technical_result
                else:  # IRRELEVANT
                    prompt = get_best_user_answer_irrelevant_prompt(
                        self.db, conversations[i], gens[a_idx], gens[b_idx])
                    model_class = BestUserAnswerIrrelevantResponse
                    extractor = get_best_user_answer_irrelevant_result

                pairwise_prompts[i].append((a_idx, b_idx, prompt, model_class, extractor))

        # Flatten prompts into a single batch
        prompts: list[str] = []
        model_classes: list[type[BaseModel]] = []
        extractors: list[Callable] = []
        for i in range(len(answers)):
            for _, _, prompt, model_class, extractor in pairwise_prompts[i]:
                prompts.append(prompt)
                model_classes.append(model_class)
                extractors.append(extractor)

        responses_per_model: list[list[BaseModel]] = []
        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(prompts, model_classes)
            model.close()
            responses_per_model.append(responses)

        # Aggregate votes
        for model_responses in responses_per_model:
            response_idx = 0
            for i in range(len(answers)):
                comparisons = pairwise_prompts[i]
                for j in range(len(comparisons)):
                    response = model_responses[response_idx]
                    a_idx, b_idx, _, _, extractor = comparisons[j]
                    winner = extractor(response)
                    if winner == 0:
                        votes[i][a_idx] += 1
                    elif winner == 1:
                        votes[i][b_idx] += 1
                    response_idx += 1

        # Select best generation per conversation
        best_generations: list[str] = []
        tournament_tracking: list[TournamentVotes] = []
        for i, gens in enumerate(answers):
            best_idx = votes[i].index(max(votes[i]))
            best_generations.append(gens[best_idx])

            # Build pairwise results for tracking
            pairwise_for_tracking: list[dict] = []
            response_idx_offset = sum(len(pairwise_prompts[j]) for j in range(i))
            for pidx_inner, (a_idx, b_idx, _, _, extractor_fn) in enumerate(pairwise_prompts[i]):
                for midx, model_responses in enumerate(responses_per_model):
                    response = model_responses[response_idx_offset + pidx_inner]
                    winner = extractor_fn(response)
                    pairwise_for_tracking.append({
                        "model": self.models[midx].model_name,
                        "candidate_a": a_idx,
                        "candidate_b": b_idx,
                        "winner": winner,
                    })

            if candidate_model_indices is not None:
                original_model_idx = candidate_model_indices[i][best_idx]
                winning_model = self.models[original_model_idx].model_name
            else:
                winning_model = self.models[best_idx].model_name if best_idx < len(self.models) else f"model_{best_idx}"
            tallies = votes[i]
            max_votes = max(tallies)
            sorted_tallies = sorted(tallies, reverse=True)
            second_max = sorted_tallies[1] if len(sorted_tallies) > 1 else 0

            global_idx = conversation_indices[i] if conversation_indices is not None else i
            tournament_tracking.append(TournamentVotes(
                question_index=global_idx,
                interaction_step=len(conversations[i].interactions) - 1,
                pairwise_results=pairwise_for_tracking,
                final_tallies=tallies,
                winning_answer_model=winning_model,
                margin=max_votes - second_max,
            ))

        return best_generations, tournament_tracking
