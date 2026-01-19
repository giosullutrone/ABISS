from typing import cast
from db_datasets.db_dataset import DBDataset
from models.model import Model
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from users.best_user_answer import BestUserAnswer
from users.prompts.user_answer_prompt import get_user_answer_prompt, UserAnswerResponse, get_user_answer_result


class UserAnswer:
    def __init__(self, 
                 db: DBDataset, 
                 models: list[Model], 
                 db_descriptions: dict[str, str]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.db_descriptions: dict[str, str] = db_descriptions
        self.best_user_answer_interaction = BestUserAnswer(db, models, db_descriptions)

    def get_user_answers(self, conversations: list[Conversation]) -> None:
        answers: list[list[str]] = [[] for _ in range(len(conversations))]
        # Get the prompts
        assert all(conversation.interactions[-1].relevance is not None for conversation in conversations), "All conversations must have a relevancy label for the last interaction."
        prompts = [
            get_user_answer_prompt(
                self.db, 
                conversation, 
                self.db_descriptions, 
                cast(RelevancyLabel, conversation.interactions[-1].relevance).value
            )
            for conversation in conversations
        ]

        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(prompts, [UserAnswerResponse] * len(prompts))
            model.close()

            for i, response in enumerate(responses):
                user_answer = get_user_answer_result(response)
                answers[i].append(user_answer)
        
        best_answers = self.best_user_answer_interaction.select_best_user_answers(conversations, answers)
        for idx, conversation in enumerate(conversations):
            conversation.interactions[-1].user_response = best_answers[idx]
