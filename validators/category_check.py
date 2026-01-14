from db_datasets.db_dataset import DBDataset
from validators.validator import Validator
from dataset_dataclasses.question import QuestionUnanswerable
from models.model import Model
from validators.prompts.category_check_prompt import get_category_validation_prompt, CategoryCheckResponse, get_category_validation_result


class CategoryCheck(Validator):
    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def validate(self, questions: list[QuestionUnanswerable]) -> list[bool]:
        valids: list[list[bool]] = [[] for _ in questions]

        for model in self.models:
            model.init()
            prompts: list[str] = []
            
            for question in questions:
                prompt = get_category_validation_prompt(self.db, question.category, question)
                prompts.append(prompt)
                if len(prompt) > 32000 * 4: # 4 chars per token approx
                    print("#"* 20, prompt, "#" * 20, flush=True, sep="\n")
            
            responses: list[str] = model.generate_batch_with_constraints(
                prompts, 
                CategoryCheckResponse.model_json_schema()
            )
            model.close()

            for i, response in enumerate(responses):
                is_valid = get_category_validation_result(response)
                valids[i].append(is_valid)

        # Majority voting across models
        final_valids: list[bool] = []
        for votes in valids:
            yes_votes = sum(votes)
            no_votes = len(votes) - yes_votes
            final_valids.append(yes_votes > no_votes)        
        return final_valids
