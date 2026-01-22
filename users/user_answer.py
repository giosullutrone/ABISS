from typing import Callable, cast
from db_datasets.db_dataset import DBDataset
from models.model import Model
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from users.best_user_answer import BestUserAnswer
from users.prompts.user_answer_prompt import (
    get_user_answer_relevant_prompt, UserAnswerRelevantResponse, get_user_answer_relevant_result,
    get_user_answer_technical_prompt, UserAnswerTechnicalResponse, get_user_answer_technical_result,
    get_user_answer_irrelevant_prompt, UserAnswerIrrelevantResponse, get_user_answer_irrelevant_result
)
from pydantic import BaseModel


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
        # Get the prompts based on relevancy classification
        assert all(conversation.interactions[-1].relevance is not None for conversation in conversations), "All conversations must have a relevancy label for the last interaction."
        
        prompts: list[str] = []
        model_classes: list[type[BaseModel]] = []
        result_extractors: list[Callable] = []
        
        for conversation in conversations:
            relevance = cast(RelevancyLabel, conversation.interactions[-1].relevance)
            
            if relevance == RelevancyLabel.RELEVANT:
                prompts.append(get_user_answer_relevant_prompt(self.db, conversation, self.db_descriptions))
                model_classes.append(UserAnswerRelevantResponse)
                result_extractors.append(get_user_answer_relevant_result)
            elif relevance == RelevancyLabel.TECHNICAL:
                prompts.append(get_user_answer_technical_prompt(self.db, conversation, self.db_descriptions))
                model_classes.append(UserAnswerTechnicalResponse)
                result_extractors.append(get_user_answer_technical_result)
            else:  # IRRELEVANT
                prompts.append(get_user_answer_irrelevant_prompt(self.db, conversation, self.db_descriptions))
                model_classes.append(UserAnswerIrrelevantResponse)
                result_extractors.append(get_user_answer_irrelevant_result)

        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(prompts, model_classes)
            model.close()

            for i, response in enumerate(responses):
                user_answer = result_extractors[i](response)
                answers[i].append(user_answer)
        
        best_answers = self.best_user_answer_interaction.select_best_user_answers(conversations, answers)
        for idx, conversation in enumerate(conversations):
            conversation.interactions[-1].user_response = best_answers[idx]
