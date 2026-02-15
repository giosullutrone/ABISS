from agents.system import System
from users.user import User
from dataset_dataclasses.question import Question
from dataset_dataclasses.benchmark import Results, Interaction, Conversation, SystemResponse
from db_datasets.db_dataset import DBDataset
from categories.category import Category
from dataset_dataclasses.benchmark import CategoryUse
from evaluators.evaluator import Evaluator
from evaluators.recognition import Recognition
from evaluators.classification import Classification
from evaluators.generation import Generation
from evaluators.feedback import Feedback


class Benchmark:
    def __init__(self, 
                 db_dataset: DBDataset, 
                 system: System, 
                 user: User, 
                 max_steps: int,
                 category_uses: list[CategoryUse]) -> None:
        self.db_dataset: DBDataset = db_dataset
        self.system: System = system
        self.user: User = user
        self.max_steps: int = max_steps
        self.category_uses: list[CategoryUse] = category_uses
        
        # Initialize evaluators automatically
        self.evaluators: list[Evaluator] = [
            Recognition(),
            Classification(),
            Generation(self.db_dataset),
            Feedback(self.db_dataset, self.user.models)
        ]

    def run(self, questions: list[Question]) -> Results:
        # Initialize conversations for each question and knowledge level
        conversations: list[Conversation] = []
        question_conversation_mapping: dict[int, list[int]] = {x: [] for x in range(len(questions))}
        for idx, question in enumerate(questions):
            conv_idx: int = idx * len(self.category_uses)
            for category_use in self.category_uses:
                conversations.append(Conversation(
                    question=question,
                    interactions=[],
                    category_use=category_use
                ))
                question_conversation_mapping[idx].append(conv_idx)
                conv_idx += 1
        
        # Step 1: Classify all questions (only once per question)        
        predicted_categories_per_question: list[Category] = self.system.get_category(
            [conversations[question_conversation_mapping[idx][0]] for idx in range(len(questions))]
        )
        # Map predicted categories back to all conversations saving them in each conversation
        for idx, conv_indices in question_conversation_mapping.items():
            for conv_idx in conv_indices:
                conversations[conv_idx].predicted_category = predicted_categories_per_question[idx]

        # Step 2: Run the interactions
        # Keep track of which conversation still hasn't received a final response (solution or feedback)
        unfinished_conversations: list[int] = list(range(len(conversations)))
        # We run the loop until all conversations are finished or we reach max steps + 1 (to allow for final feedback)
        step = 0
        while len(unfinished_conversations) > 0 and step < self.max_steps + 1:
            # Get system responses for unfinished conversations
            # First we prepare the predicted categories:
            # - If category_use is NO_CATEGORY, we pass None
            # - If category_use is PREDICTED, we pass the predicted category
            # - If category_use is GROUND_TRUTH, we pass the ground truth category from the question
            categories_to_pass: list[Category | None] = []
            for idx in unfinished_conversations:
                conv = conversations[idx]
                if conv.category_use == CategoryUse.NO_CATEGORY:
                    categories_to_pass.append(None)
                elif conv.category_use == CategoryUse.PREDICTED:
                    categories_to_pass.append(conv.predicted_category)
                else:
                    categories_to_pass.append(conv.question.category)

            system_responses: list[SystemResponse] = self.system.get_system_response(
                [conversations[idx] for idx in unfinished_conversations],
                categories_to_pass,
                [step for _ in unfinished_conversations]
            )

            # Process each system response
            # First we create an Interaction for each conversation with the system response
            for i, idx in enumerate(unfinished_conversations):
                conv = conversations[idx]
                sys_resp = system_responses[i]
                interaction = Interaction(system_response=sys_resp)
                conv.interactions.append(interaction)
            
            # Second we update the predicted SQL and the feedback if the system provided one
            # and stop the conversations that received a final response
            indices_to_remove: list[int] = []
            for i, idx in enumerate(unfinished_conversations):
                conv = conversations[idx]
                sys_resp = system_responses[i]
                if sys_resp.system_sql is not None:
                    conv.predicted_sql = sys_resp.system_sql
                    indices_to_remove.append(idx)
            
                if sys_resp.system_feedback is not None:
                    conv.predicted_feedback = sys_resp.system_feedback
                    indices_to_remove.append(idx)
            
            # Remove finished conversations
            for idx in indices_to_remove:
                if idx in unfinished_conversations:
                    unfinished_conversations.remove(idx)

            # Now we add the relevancy labels and user answers for the remaining conversations 
            # (those that have received a question and are still ongoing)
            if len(unfinished_conversations) > 0:
                self.user.get_relevancy(
                    [conversations[idx] for idx in unfinished_conversations]
                )

                # Now we get user answers
                self.user.get_answers(
                    [conversations[idx] for idx in unfinished_conversations]
                )

            # Increment step
            step += 1

        # Create results object
        results = Results(
            agent_name=self.system.agent_name,
            user_name=self.user.agent_name,
            dataset_name=self.db_dataset.db_name,
            conversations=conversations
        )
        
        # Run evaluators automatically
        self.evaluate(results)
        
        return results
    
    def evaluate(self, results: Results) -> None:
        for evaluator in self.evaluators:
            evaluator.evaluate(results.conversations)
