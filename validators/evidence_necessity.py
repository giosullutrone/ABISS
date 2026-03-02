from tqdm import tqdm
from validators.validator import Validator
from dataset_dataclasses.question import Question
from models.model import Model
from db_datasets.db_dataset import DBDataset
from dataset_dataclasses.council_tracking import ValidationStageResult, QuestionVotes, ModelVote


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

    def validate(self, questions: list[Question]) -> ValidationStageResult:
        valids: list[bool] = [True for _ in questions]

        # Generate SQLs for each question WITHOUT evidence using all models
        sqls_per_model: list[list[str | None]] = [
            self.db.generate_sqls_without_evidence_unsafe(model, questions)
            for model in self.models
        ]

        # For each question, check if a majority of models can produce equivalent SQL without evidence
        question_votes: list[QuestionVotes] = []
        for i, question in tqdm(enumerate(questions), total=len(questions), desc="Comparing SQL results (evidence necessity)"):
            gt_sql = question.sql
            assert gt_sql is not None, "GT SQL query is None for answerable question."

            per_model_votes: list[ModelVote] = []
            equivalent_count = 0
            for midx, model_sqls in enumerate(sqls_per_model):
                sql = model_sqls[i]
                if sql is None:
                    per_model_votes.append(ModelVote(model_name=self.models[midx].model_name, vote=False))
                    continue

                is_equivalent = self.db.compare_query_results(
                    db_id=question.db_id,
                    predicted_sql=sql,
                    ground_truth_sql=gt_sql
                )
                per_model_votes.append(ModelVote(model_name=self.models[midx].model_name, vote=is_equivalent))
                if is_equivalent:
                    equivalent_count += 1

            if equivalent_count > len(self.models) / 2:
                valids[i] = False

            question_votes.append(QuestionVotes(
                question_index=i,
                question_text=question.question,
                votes=per_model_votes,
                aggregate_result=valids[i],
                removed=not valids[i],
            ))

        return ValidationStageResult(
            stage_name="evidence_necessity",
            validities=valids,
            question_votes=question_votes,
        )
