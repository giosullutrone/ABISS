from db_datasets.db_dataset import DBDataset
from models.model import Model
from dataset_dataclasses.results import Conversation
from interactions.best_user_answer import BestUserAnswer
from prompts.user_answer_prompt import get_user_answer_prompt, UserAnswerResponse, get_user_answer_result


class UserAnswer:
    """
    Module that given a list of QuestionUnanswerable and a list of models, generates the answer provided by the user to the clarification question asked by the text-to-SQL system.
    The answer is expected to help disambiguate the original question. It may be question related to the hidden knowledge or may be technical question related to SQL aspects (like ordering or limits).
    
    We return a list of answers for each question, one for each model, to be used in a 1vs1 voting scheme later on.
    """

    def __init__(self, 
                 db: DBDataset, 
                 models: list[Model], 
                 db_descriptions: dict[str, str]) -> None:
        self.db: DBDataset = db
        self.models: list[Model] = models
        self.db_descriptions: dict[str, str] = db_descriptions
        self.best_user_answer_interaction = BestUserAnswer(db, models, db_descriptions)

    def get_user_answers(self, conversations: list[Conversation]) -> list[Conversation]:
        answers: list[list[str]] = [[] for _ in range(len(conversations))]
        # Get the prompts
        prompts = [get_user_answer_prompt(self.db, conversation, conversation.user_knowledge_level, self.db_descriptions) for conversation in conversations]

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
        return conversations
