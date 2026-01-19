from db_datasets.db_dataset import DBDataset
from models.model import Model
from dataset_dataclasses.benchmark import Conversation, RelevancyLabel
from users.prompts.question_relevancy_prompt import get_relevancy_prompt, QuestionRelevancyResponse, get_question_relevancy_result


class QuestionRelevancy:
    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def get_relevancy(self, conversations: list[Conversation]) -> None:
        labels: list[dict[RelevancyLabel, int]] = [{RelevancyLabel.RELEVANT: 0, RelevancyLabel.TECHNICAL: 0, RelevancyLabel.IRRELEVANT: 0} for _ in range(len(conversations))]

        # Get the prompts
        prompts = [get_relevancy_prompt(conversation) for conversation in conversations]
        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints(prompts, [QuestionRelevancyResponse] * len(prompts))
            model.close()

            for i, response in enumerate(responses):
                relevancy_label = get_question_relevancy_result(response)
                labels[i][relevancy_label] += 1
                # If parsing fails, default to Irrelevant (no increment)
        
        final_labels: list[RelevancyLabel] = []
        for label_counts in labels:
            sorted_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
            # Get the top two labels
            top_label, top_count = sorted_labels[0]
            _, second_count = sorted_labels[1]
            # If there's a tie for the top label, choose Irrelevant
            if top_count == second_count:
                final_labels.append(RelevancyLabel.IRRELEVANT)
            else:
                final_labels.append(top_label)
        
        for i, conversation in enumerate(conversations):
            conversation.interactions[-1].relevance = final_labels[i]
