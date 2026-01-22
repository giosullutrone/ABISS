from typing import cast, Callable
from pydantic import BaseModel
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from models.model import Model
from db_datasets.db_dataset import DBDataset
from users.prompts.best_user_answer_prompt import (
    get_best_user_answer_relevant_prompt, BestUserAnswerRelevantResponse, get_best_user_answer_relevant_result,
    get_best_user_answer_technical_prompt, BestUserAnswerTechnicalResponse, get_best_user_answer_technical_result,
    get_best_user_answer_irrelevant_prompt, BestUserAnswerIrrelevantResponse, get_best_user_answer_irrelevant_result
)


class BestUserAnswer:
    """
    Interaction that selects the best user answer generated among multiple candidates based on a 1vs1 comparison performed by a list of models on all candidates.
    """

    def __init__(self, db: DBDataset, models: list[Model], db_descriptions: dict[str, str]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.db_descriptions: dict[str, str] = db_descriptions

    def select_best_user_answers(self, conversations: list[Conversation], answers: list[list[str]]) -> list[str]:
        """
        Given a dictionary of db ids and candidate user answers, select the best one (for each db) based on 1vs1 comparisons by the models.
        """
        # Do the operation one model at a time (this way we don't have to unload and load the weights multiple times)
        votes: list[list[int]] = [[0] * len(ans) for ans in answers]

        # Generate all the pairwise comparison prompts with appropriate prompt for each relevancy type
        pairwise_prompts: dict[int, list[tuple[int, int, str, type[BaseModel], Callable]]] = {}
        for i, gens in enumerate(answers):
            pairwise_prompts[i] = []
            assert conversations[i].interactions[-1].relevance is not None, "Relevancy label must be set for the last interaction."
            relevance = cast(RelevancyLabel, conversations[i].interactions[-1].relevance)
            
            for j in range(len(gens)):
                for k in range(len(gens)):
                    if j != k:
                        if relevance == RelevancyLabel.RELEVANT:
                            prompt = get_best_user_answer_relevant_prompt(self.db, self.db_descriptions, conversations[i], gens[j], gens[k])
                            model_class = BestUserAnswerRelevantResponse
                            extractor = get_best_user_answer_relevant_result
                        elif relevance == RelevancyLabel.TECHNICAL:
                            prompt = get_best_user_answer_technical_prompt(self.db, self.db_descriptions, conversations[i], gens[j], gens[k])
                            model_class = BestUserAnswerTechnicalResponse
                            extractor = get_best_user_answer_technical_result
                        else:  # IRRELEVANT
                            prompt = get_best_user_answer_irrelevant_prompt(self.db, self.db_descriptions, conversations[i], gens[j], gens[k])
                            model_class = BestUserAnswerIrrelevantResponse
                            extractor = get_best_user_answer_irrelevant_result
                        
                        pairwise_prompts[i].append((j, k, prompt, model_class, extractor))

        # Now flatten the prompts into a single list
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
        
        # Now aggregate the votes
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

        # Now select the best generation for each db based on the votes
        best_generations: list[str] = []
        for i, gens in enumerate(answers):
            best_idx = votes[i].index(max(votes[i]))
            best_generations.append(gens[best_idx])
        return best_generations