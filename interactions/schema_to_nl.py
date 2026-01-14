from db_datasets.db_dataset import DBDataset
from models.model import Model
from interactions.best_description import BestDescription
from interactions.prompts.schema_to_nl_prompt import get_generation_prompt, SchemaToNLResponse, get_schema_to_nl_result


class SchemaToNL:
    """
    Module that given a list of database schemas and models, generates a natural language detailed description of each schema.
    """

    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.best_description_interaction: BestDescription = BestDescription(db, models)

    def generate_descriptions(self, db_ids: list[str]) -> dict[str, str]:
        """
        Generate natural language descriptions for the given database schemas.
        """
        all_descriptions: dict[str, list[str]] = {db_id: [] for db_id in db_ids}

        for model in self.models:
            model.init()
            responses = model.generate_batch_with_constraints([get_generation_prompt(self.db, db_id) for db_id in db_ids], SchemaToNLResponse.model_json_schema())
            model.close()

            for i, response in enumerate(responses):
                description = get_schema_to_nl_result(response)
                all_descriptions[db_ids[i]].append(description)

        descriptions = self.best_description_interaction.select_best_descriptions(all_descriptions)
        return descriptions
