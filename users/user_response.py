"""Two-stage user response: Stage 1 classifies and locates AST nodes,
Stage 2 generates natural-language-only answers grounded in matched source.

Replaces the previous single-stage pipeline. Following BIRD-Interact's
(Li et al., 2025) function-driven approach adapted for ABISS.
"""

from typing import Callable
from db_datasets.db_dataset import DBDataset
from models.model import Model
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.question import QuestionUnanswerable
from dataset_dataclasses.council_tracking import RelevancyVotes, TournamentVotes
from users.best_user_answer import BestUserAnswer
from users.sql_ast import parse_sql_to_nodes, SQLNode
from users.prompts.user_classify_prompt import (
    UserClassifySolvable,
    UserClassifyAnswerable,
    get_user_classify_prompt_solvable,
    get_user_classify_prompt_answerable,
    get_classify_solvable_result,
    get_classify_answerable_result,
)
from users.prompts.user_answer_prompt import (
    UserAnswerModel,
    get_user_answer_prompt_relevant,
    get_user_answer_prompt_technical,
    get_user_answer_prompt_irrelevant,
)
from pydantic import BaseModel


class UserResponse:
    """Two-stage user response pipeline."""

    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db = db
        self.models = models
        self.best_user_answer = BestUserAnswer(db, models)

    def get_response(self, conversations: list[Conversation]) -> tuple[list[RelevancyVotes], list[TournamentVotes]]:
        # ---- short-circuit unsolvable questions ----
        to_process: list[int] = []

        for i, conv in enumerate(conversations):
            is_solvable = (
                conv.question.category.is_solvable()
                if isinstance(conv.question, QuestionUnanswerable)
                else True
            )

            if not is_solvable:
                conv.interactions[-1].relevance = RelevancyLabel.IRRELEVANT
                conv.interactions[-1].user_response = "That's not relevant to my question."
                continue

            to_process.append(i)

        if not to_process:
            return [], []

        # ---- pre-parse GT SQL into AST nodes ----
        nodes_per_conv: list[list[SQLNode]] = []
        for idx in to_process:
            sql = conversations[idx].question.sql
            nodes_per_conv.append(parse_sql_to_nodes(sql))

        # ---- STAGE 1: Classification + Source Identification ----
        stage1_prompts: list[str] = []
        stage1_model_classes: list[type[BaseModel]] = []
        stage1_extractors: list[Callable] = []

        for pidx, conv_idx in enumerate(to_process):
            conv = conversations[conv_idx]
            nodes = nodes_per_conv[pidx]
            is_answerable = conv.question.category.is_answerable()

            if is_answerable:
                stage1_prompts.append(get_user_classify_prompt_answerable(conv, nodes))
                stage1_model_classes.append(UserClassifyAnswerable)
                stage1_extractors.append(get_classify_answerable_result)
            else:
                stage1_prompts.append(get_user_classify_prompt_solvable(conv, nodes))
                stage1_model_classes.append(UserClassifySolvable)
                stage1_extractors.append(get_classify_solvable_result)

        # Collect per-model classifications
        relevancy_votes: list[dict[RelevancyLabel, int]] = [
            {RelevancyLabel.RELEVANT: 0, RelevancyLabel.TECHNICAL: 0, RelevancyLabel.IRRELEVANT: 0}
            for _ in stage1_prompts
        ]
        per_model_node_ids: list[list[list[int]]] = [[] for _ in stage1_prompts]
        per_model_labels: list[list[RelevancyLabel]] = [[] for _ in stage1_prompts]

        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(stage1_prompts, stage1_model_classes)
            model.close()

            for pidx, response in enumerate(responses):
                label, node_ids = stage1_extractors[pidx](response)
                relevancy_votes[pidx][label] += 1
                per_model_labels[pidx].append(label)
                per_model_node_ids[pidx].append(node_ids)

        # ---- Majority vote on labels ----
        final_labels: list[RelevancyLabel] = []
        for pidx in range(len(stage1_prompts)):
            votes = relevancy_votes[pidx]
            sorted_labels = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
            top_label, top_count = sorted_labels[0]
            _, second_count = sorted_labels[1]
            if top_count == second_count:
                final_labels.append(RelevancyLabel.IRRELEVANT)
            else:
                final_labels.append(top_label)

        # ---- Resolve sources based on winning label ----
        resolved_fragments: list[list[str]] = []
        for pidx in range(len(stage1_prompts)):
            if final_labels[pidx] == RelevancyLabel.TECHNICAL:
                union_ids: set[int] = set()
                for midx in range(len(self.models)):
                    if per_model_labels[pidx][midx] == RelevancyLabel.TECHNICAL:
                        union_ids.update(per_model_node_ids[pidx][midx])
                nodes = nodes_per_conv[pidx]
                node_map = {n.node_id: n.sql_fragment for n in nodes}
                fragments = [node_map[nid] for nid in sorted(union_ids) if nid in node_map]
                resolved_fragments.append(fragments)
            else:
                resolved_fragments.append([])

        # ---- Build relevancy tracking ----
        relevancy_tracking: list[RelevancyVotes] = []
        for pidx, conv_idx in enumerate(to_process):
            model_labels = [
                (self.models[midx].model_name, per_model_labels[pidx][midx].value)
                for midx in range(len(self.models))
            ]
            all_same = len(set(label for _, label in model_labels)) == 1
            relevancy_tracking.append(RelevancyVotes(
                question_index=conv_idx,
                interaction_step=len(conversations[conv_idx].interactions) - 1,
                per_model_labels=model_labels,
                winning_label=final_labels[pidx].value,
                unanimous=all_same,
            ))

        # ---- Assign labels ----
        for pidx, conv_idx in enumerate(to_process):
            conversations[conv_idx].interactions[-1].relevance = final_labels[pidx]

        # ---- STAGE 2: Response Generation ----
        stage2_prompts: list[str] = []
        for pidx, conv_idx in enumerate(to_process):
            conv = conversations[conv_idx]
            label = final_labels[pidx]
            if label == RelevancyLabel.RELEVANT:
                stage2_prompts.append(get_user_answer_prompt_relevant(conv))
            elif label == RelevancyLabel.TECHNICAL:
                stage2_prompts.append(get_user_answer_prompt_technical(conv, resolved_fragments[pidx]))
            else:
                stage2_prompts.append(get_user_answer_prompt_irrelevant(conv))

        stage2_model_classes = [UserAnswerModel] * len(stage2_prompts)

        per_model_answers: list[list[str]] = [[] for _ in stage2_prompts]
        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(stage2_prompts, stage2_model_classes)
            model.close()

            for pidx, response in enumerate(responses):
                validated = UserAnswerModel.model_validate(response)
                per_model_answers[pidx].append(validated.answer.strip())

        # ---- Select best answer via tournament ----
        best_answers: list[str] = []
        needs_tournament_indices: list[int] = []
        needs_tournament_convs: list[Conversation] = []
        needs_tournament_answers: list[list[str]] = []
        needs_tournament_fragments: list[list[str]] = []

        for pidx in range(len(stage2_prompts)):
            answers = per_model_answers[pidx]
            if len(answers) == 1 or len(set(answers)) == 1:
                # Single answer or all answers identical: skip tournament (any would win).
                best_answers.append(answers[0])
            else:
                best_answers.append("")  # placeholder
                needs_tournament_indices.append(pidx)
                needs_tournament_convs.append(conversations[to_process[pidx]])
                needs_tournament_answers.append(answers)
                needs_tournament_fragments.append(resolved_fragments[pidx])

        tournament_tracking: list[TournamentVotes] = []
        if needs_tournament_convs:
            tournament_results, tournament_tracking = self.best_user_answer.select_best_user_answers(
                needs_tournament_convs, needs_tournament_answers,
                sql_fragments_per_conv=needs_tournament_fragments,
            )
            for i, pidx in enumerate(needs_tournament_indices):
                best_answers[pidx] = tournament_results[i]

        for pidx, conv_idx in enumerate(to_process):
            conversations[conv_idx].interactions[-1].user_response = best_answers[pidx]

        return relevancy_tracking, tournament_tracking
