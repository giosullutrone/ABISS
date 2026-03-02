from db_datasets.db_dataset import DBDataset
from validators.validator import Validator
from dataset_dataclasses.question import Question
from models.model import Model
from categories.category import Category
from validators.prompts.category_consistency_prompt import get_category_consistency_prompt, CategoryConsistencyResponse, get_category_consistency_result
from pydantic import BaseModel
from typing import cast
from dataset_dataclasses.council_tracking import ValidationStageResult, QuestionVotes, ModelVote


class CategoryConsistency(Validator):
    def __init__(self, db: DBDataset, models: list[Model], categories: list[Category]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.categories: list[Category] = categories

    def validate(self, questions: list[Question]) -> ValidationStageResult:
        # For each question, compare its main category against all other categories
        # Use pairwise comparisons, testing both orders
        
        # Build prompts for all comparisons
        prompts: list[str] = []
        prompt_metadata: list[tuple[int, int, bool]] = []  # (question_idx, other_cat_idx, is_main_first)
        
        for q_idx, question in enumerate(questions):
            main_cat = question.category
            for cat_idx, other_cat in enumerate(self.categories):
                if other_cat == main_cat:
                    continue
                # Only compare against categories of the same type:
                # - ambiguous (is_solvable=True)
                # - unanswerable (is_solvable=False)
                # Answerable questions are excluded upstream (generator.py).
                if other_cat.is_answerable() or main_cat.is_solvable() != other_cat.is_solvable():
                    continue
                
                # Prompt with main as A, other as B
                prompt = get_category_consistency_prompt(self.db, main_cat, other_cat, question)
                prompts.append(prompt)
                prompt_metadata.append((q_idx, cat_idx, True))
                
                # Prompt with other as A, main as B
                prompt = get_category_consistency_prompt(self.db, other_cat, main_cat, question)
                prompts.append(prompt)
                prompt_metadata.append((q_idx, cat_idx, False))
        
        # Collect votes from all models
        votes_per_question_other: dict[tuple[int, int], list[int]] = {}  # key: (q_idx, cat_idx), value: list of 0 (main wins) or 1 (other wins)
        
        for model in self.models:
            model.init()
            responses: list[BaseModel | None] = model.generate_batch_with_constraints_unsafe(prompts, cast(list[type[BaseModel]], [CategoryConsistencyResponse] * len(prompts)))
            model.close()
            
            # Process responses
            for i, response in enumerate(responses):
                if response is None:
                    winner = 1  # Assume other wins if response is None
                else:
                    winner = get_category_consistency_result(response)  # 0 for A, 1 for B
                q_idx, cat_idx, is_main_first = prompt_metadata[i]
                if is_main_first:
                    # A is main, B is other
                    if winner == 0:
                        vote = 0  # main wins
                    else:
                        vote = 1  # other wins
                else:
                    # A is other, B is main
                    if winner == 0:
                        vote = 1  # other wins
                    else:
                        vote = 0  # main wins
                
                key = (q_idx, cat_idx)
                if key not in votes_per_question_other:
                    votes_per_question_other[key] = []
                votes_per_question_other[key].append(vote)
        
        # For each question, check if main category wins majority against all other categories
        final_valids: list[bool] = []
        question_votes: list[QuestionVotes] = []
        for q_idx in range(len(questions)):
            main_wins_all = True
            per_comparison_votes: list[ModelVote] = []
            for cat_idx in range(len(self.categories)):
                if self.categories[cat_idx] == questions[q_idx].category:
                    continue

                key = (q_idx, cat_idx)
                if key in votes_per_question_other:
                    votes = votes_per_question_other[key]
                    main_votes = sum(1 for v in votes if v == 0)
                    other_votes = len(votes) - main_votes
                    for vidx, v in enumerate(votes):
                        model_idx = vidx // 2
                        model_name = self.models[model_idx].model_name if model_idx < len(self.models) else f"model_{model_idx}"
                        per_comparison_votes.append(ModelVote(
                            model_name=f"{model_name}_vs_{self.categories[cat_idx].get_name()}",
                            vote=v,
                        ))
                    if main_votes <= other_votes:
                        main_wins_all = False
                        print(f"Question {q_idx} does not prefer main category over {self.categories[cat_idx].get_name()}.")
                        print(f"Question: {questions[q_idx].question}, Main: {questions[q_idx].category.get_name()}, Other: {self.categories[cat_idx].get_name()}, Main votes: {main_votes}, Other votes: {other_votes}")
                        break

            final_valids.append(main_wins_all)
            question_votes.append(QuestionVotes(
                question_index=q_idx,
                question_text=questions[q_idx].question,
                votes=per_comparison_votes,
                aggregate_result=main_wins_all,
                removed=not main_wins_all,
            ))

        return ValidationStageResult(
            stage_name="category_consistency",
            validities=final_valids,
            question_votes=question_votes,
        )
