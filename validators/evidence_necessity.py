from validators.validator import Validator
from dataset_dataclasses.question import Question
from models.model import Model
from db_datasets.db_dataset import DBDataset


class EvidenceNecessity(Validator):
    """
    Validates that evidence is truly necessary for answerable-with-evidence questions.

    The check works by asking all models in the council to generate SQL for the question
    WITHOUT providing the evidence. If a majority of models are able to produce an SQL
    query whose results are equivalent to the ground truth SQL, the evidence is deemed
    unnecessary and the question is invalid. Using majority rather than "any" avoids
    discarding questions based on a single model's lucky guess or pre-training bias.
    """

    def __init__(self, db: DBDataset, models: list[Model]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models

    def validate(self, questions: list[Question]) -> list[bool]:
        valids: list[bool] = [True for _ in questions]

        # Generate SQLs for each question WITHOUT evidence using all models
        sqls_per_model: list[list[str | None]] = [
            self.db.generate_sqls_without_evidence_unsafe(model, questions)
            for model in self.models
        ]

        # For each question, check if a majority of models can produce equivalent SQL without evidence
        for i, question in enumerate(questions):
            gt_sql = question.sql
            assert gt_sql is not None, "GT SQL query is None for answerable question."

            equivalent_count = 0
            for model_sqls in sqls_per_model:
                sql = model_sqls[i]
                if sql is None:
                    continue

                if self.db.compare_query_results(
                    db_id=question.db_id,
                    predicted_sql=sql,
                    ground_truth_sql=gt_sql
                ):
                    equivalent_count += 1

            # If a majority of models can produce equivalent SQL without evidence,
            # the evidence is not necessary — mark as invalid
            # (ties resolve conservatively: evidence is deemed necessary)
            if equivalent_count > len(self.models) / 2:
                valids[i] = False

        return valids
