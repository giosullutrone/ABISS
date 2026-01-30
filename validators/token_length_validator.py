from validators.validator import Validator
from dataset_dataclasses.question import Question, QuestionUnanswerable
from models.model import Model


class TokenLengthValidator(Validator):
    def __init__(self, max_length: int, models: list[Model]):
        self.max_length = max_length
        self.models = models

    def validate(self, questions: list[Question]) -> list[bool]:
        # Prepare concatenated prompts for all questions
        concatenated_prompts = []
        for question in questions:
            # Concatenate question, sql, hidden_knowledge, evidence
            hidden_knowledge = ""
            if isinstance(question, QuestionUnanswerable):
                hidden_knowledge = question.hidden_knowledge or ""
            
            parts = [
                question.question,
                question.sql or "",
                hidden_knowledge,
                question.evidence or ""
            ]
            concatenated = " ".join(parts).strip()
            concatenated_prompts.append(concatenated)
        
        # Get token lengths from all models in batch
        model_token_lengths = []
        for model in self.models:
            model.init()
            lengths = model.get_token_lengths(concatenated_prompts)
            model_token_lengths.append(lengths)
            model.close()
        
        # For each question, check if max token length across models <= max_length
        results = []
        for i in range(len(questions)):
            max_tokens = max(lengths[i] for lengths in model_token_lengths) if model_token_lengths else 0
            results.append(max_tokens <= self.max_length)
        
        return results