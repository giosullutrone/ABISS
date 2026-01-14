from pydantic import BaseModel
from dataset_dataclasses.results import Conversation
from models.model import Model
from prompts import UserKnowledgeLevel, UserAnswerStyle
from db_datasets.db_dataset import DBDataset
from prompts.best_user_answer_prompt import get_selection_prompt, BestUserAnswerResponse, get_best_user_answer_result


class BestUserAnswer:
    """
    Interaction that selects the best user answer generated among multiple candidates based on a 1vs1 comparison performed by a list of models on all candidates.
    """

    def __init__(self, db: DBDataset, models: list[Model], db_descriptions: dict[str, str] | None, user_answer_style: UserAnswerStyle, user_knowledge_level: UserKnowledgeLevel) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.db_descriptions: dict[str, str] | None = db_descriptions
        self.user_answer_style: UserAnswerStyle = user_answer_style
        self.user_knowledge_level: UserKnowledgeLevel = user_knowledge_level

    def select_best_user_answers(self, conversations: list[Conversation], answers: list[list[str]]) -> list[str]:
        """
        Given a dictionary of db ids and candidate user answers, select the best one (for each db) based on 1vs1 comparisons by the models.
        """
        # Do the operation one model at a time (this way we don't have to unload and load the weights multiple times)
        votes: list[list[int]] = [[0] * len(ans) for ans in answers]

        # Generate all the pairwise comparison prompts
        pairwise_prompts: dict[int, list[tuple[int, int, str]]] = {}
        for i, gens in enumerate(answers):
            pairwise_prompts[i] = []
            for j in range(len(gens)):
                for k in range(len(gens)):
                    if j != k:
                        prompt = get_selection_prompt(self.db, self.db_descriptions, conversations[i], gens[j], gens[k], self.user_knowledge_level, self.user_answer_style)
                        pairwise_prompts[i].append((j, k, prompt))

        # Now flatten the prompts into a single list
        prompts: list[str] = []
        for i in range(len(answers)):
            prompts.extend([prompt for _, _, prompt in pairwise_prompts[i]])

        responses_per_model: list[list[BaseModel]] = []
        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(prompts, [BestUserAnswerResponse] * len(prompts))
            model.close()
            responses_per_model.append(responses)
        
        # Now aggregate the votes
        for model_responses in responses_per_model:
            response_idx = 0
            for i in range(len(answers)):
                comparisons = pairwise_prompts[i]
                for j in range(len(comparisons)):
                    response = model_responses[response_idx]
                    winner = get_best_user_answer_result(response)
                    a_idx, b_idx, _ = comparisons[j]
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