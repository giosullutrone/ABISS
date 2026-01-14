from models.model import Model
from db_datasets.db_dataset import DBDataset
from interactions.prompts.best_description_prompt import get_selection_prompt, BestDescriptionResponse, get_best_description_result


class BestDescription:
    """
    Interaction that selects the best description generated among multiple candidates based on a 1vs1 comparison performed by a list of models on all candidates.
    """

    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def select_best_descriptions(self, descriptions: dict[str, list[str]]) -> dict[str, str]:
        """
        Given a dictionary of db ids and candidate descriptions, select the best one (for each db) based on 1vs1 comparisons by the models.
        """
        # Do the operation one model at a time (this way we don't have to unload and load the weights multiple times)
        votes: dict[str, list[int]] = {db_id: [0] * len(gens) for db_id, gens in descriptions.items()}

        # Generate all the pairwise comparison prompts
        pairwise_prompts: dict[str, list[tuple[int, int, str]]] = {}
        for db_id, gens in descriptions.items():
            pairwise_prompts[db_id] = []
            for i in range(len(gens)):
                for j in range(len(gens)):
                    if i != j:
                        prompt = get_selection_prompt(self.db, db_id, gens[i], gens[j])
                        pairwise_prompts[db_id].append((i, j, prompt))

        # Now flatten the prompts into a single list
        prompts: list[str] = []
        for db_id in descriptions.keys():
            prompts.extend([prompt for _, _, prompt in pairwise_prompts[db_id]])

        responses_per_model: list[list[str]] = []
        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(prompts, BestDescriptionResponse.model_json_schema())
            model.close()
            responses_per_model.append(responses)

        # Now aggregate the votes
        for model_responses in responses_per_model:
            response_idx = 0
            for db_id in descriptions.keys():
                comparisons = pairwise_prompts[db_id]
                for j in range(len(comparisons)):
                    response = model_responses[response_idx]
                    winner = get_best_description_result(response)
                    a_idx, b_idx, _ = comparisons[j]
                    if winner == 0:
                        votes[db_id][a_idx] += 1
                    elif winner == 1:
                        votes[db_id][b_idx] += 1
                    response_idx += 1

        # Now select the best generation for each db based on the votes
        best_generations: dict[str, str] = {}
        for db_id, gens in descriptions.items():
            best_idx = votes[db_id].index(max(votes[db_id]))
            best_generations[db_id] = gens[best_idx]
        return best_generations
