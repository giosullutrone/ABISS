from db_datasets.db_dataset import DBDataset
from models.model import Model
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from dataset_dataclasses.question import QuestionUnanswerable
from users.prompts.question_relevancy_prompt import (
    get_relevancy_prompt_answerable, QuestionRelevancyAnswerableResponse, get_question_relevancy_answerable_result,
    get_relevancy_prompt_solvable, QuestionRelevancySolvableResponse, get_question_relevancy_solvable_result
)
from typing import Callable


class QuestionRelevancy:
    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def get_relevancy(self, conversations: list[Conversation]) -> None:
        labels: list[dict[RelevancyLabel, int]] = [{RelevancyLabel.RELEVANT: 0, RelevancyLabel.TECHNICAL: 0, RelevancyLabel.IRRELEVANT: 0} for _ in range(len(conversations))]

        # Separate conversations by question type to apply appropriate relevancy logic
        conversations_to_classify: list[int] = []  # Indices of conversations needing classification
        prompts: list[str] = []
        model_classes: list[type] = []
        result_extractors: list[Callable] = []
        
        for i, conversation in enumerate(conversations):
            is_answerable = conversation.question.category.is_answerable()
            is_solvable = conversation.question.category.is_solvable() if isinstance(conversation.question, QuestionUnanswerable) else True
            
            if not is_solvable:
                # Unsolvable: Skip relevancy check, assign IRRELEVANT directly
                conversation.interactions[-1].relevance = RelevancyLabel.IRRELEVANT
            elif is_answerable:
                # Answerable: Can only be TECHNICAL or IRRELEVANT (use answerable prompt)
                conversations_to_classify.append(i)
                prompts.append(get_relevancy_prompt_answerable(conversation))
                model_classes.append(QuestionRelevancyAnswerableResponse)
                result_extractors.append(get_question_relevancy_answerable_result)
            else:
                # Solvable: Can be RELEVANT, TECHNICAL, or IRRELEVANT (use solvable prompt)
                conversations_to_classify.append(i)
                prompts.append(get_relevancy_prompt_solvable(conversation))
                model_classes.append(QuestionRelevancySolvableResponse)
                result_extractors.append(get_question_relevancy_solvable_result)
        
        if len(prompts) == 0:
            return  # All conversations were unsolvable
        
        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(prompts, model_classes)
            model.close()

            for prompt_idx, response in enumerate(responses):
                conv_idx = conversations_to_classify[prompt_idx]
                conversation = conversations[conv_idx]
                relevancy_label = result_extractors[prompt_idx](response)
                
                # Note: Constraints are already enforced by using the appropriate prompt and response model
                # Answerable uses QuestionRelevancyAnswerableResponse (only Technical/Irrelevant)
                # Solvable uses QuestionRelevancySolvableResponse (Relevant/Technical/Irrelevant)
                
                labels[conv_idx][relevancy_label] += 1
        
        final_labels: list[RelevancyLabel] = []
        for i, label_counts in enumerate(labels):
            if conversations[i].interactions[-1].relevance is not None:
                # Already assigned (unsolvable)
                continue
                
            sorted_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
            # Get the top two labels
            top_label, top_count = sorted_labels[0]
            _, second_count = sorted_labels[1]
            # If there's a tie for the top label, choose Irrelevant
            if top_count == second_count:
                final_labels.append(RelevancyLabel.IRRELEVANT)
            else:
                final_labels.append(top_label)
        
        # Assign final labels to conversations that needed classification
        label_idx = 0
        for i, conversation in enumerate(conversations):
            if conversation.interactions[-1].relevance is None:
                conversation.interactions[-1].relevance = final_labels[label_idx]
                label_idx += 1
