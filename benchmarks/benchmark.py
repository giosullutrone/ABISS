from agents.system import System
from users.user import User
from dataset_dataclasses.question import Question
from dataset_dataclasses.benchmark import Results, Interaction, Conversation, SystemResponse
from db_datasets.db_dataset import DBDataset
from categories.category import Category
from dataset_dataclasses.benchmark import UserKnowledgeLevel, CategoryUse
from evaluators.evaluator import Evaluator
from evaluators.recognition import Recognition
from evaluators.classification import Classification
from evaluators.generation import Generation
from evaluators.feedback import Feedback


class Benchmark:
    """
    Benchmark for evaluating text-to-SQL systems with clarification interactions.
    
    The benchmark runs conversations between a system and a simulated user, where:
    1. The system classifies questions into categories (answerable/unanswerable-solvable/unsolvable)
    2. The system asks clarification questions or provides SQL/feedback
    3. The user responds based on relevancy and knowledge level
    
    Each question is tested with different combinations of:
    - User knowledge levels (FULL/NL/NONE)
    - Category usage (GROUND_TRUTH/PREDICTED/NO_CATEGORY)
    
    After running the benchmark, use evaluators to assess performance:
    - Recognition: Check if system identified answerable vs unanswerable correctly
    - Classification: Check if system identified the exact category correctly  
    - Generation: Check if predicted SQL matches ground truth
    - Feedback: Check if system's feedback for unsolvable questions is correct
    
    Evaluators are initialized automatically and run after all conversations complete.
    """
    def __init__(self, 
                 db_dataset: DBDataset, 
                 system: System, 
                 user: User, 
                 max_steps: int) -> None:
        self.db_dataset: DBDataset = db_dataset
        self.system: System = system
        self.user: User = user
        self.max_steps: int = max_steps
        
        # Initialize evaluators automatically
        self.evaluators: list[Evaluator] = [
            Recognition(),
            Classification(),
            Generation(self.db_dataset),
            Feedback(self.db_dataset, self.user.models)
        ]

    def run(self, questions: list[Question]) -> Results:
        """
        Run the benchmark on a list of questions.
        
        Flow:
        1. Create conversations for each question × knowledge_level × category_use combination
        2. Classify all questions once (get predicted categories)
        3. For each step (0 to max_steps):
           a. System generates response (question, SQL, or feedback)
           b. Add interaction to conversation
           c. Mark conversations as finished if they received SQL or feedback
           d. For unfinished conversations with questions:
              - Get relevancy labels (Relevant/Technical/Irrelevant)
              - Get user answers based on relevancy and knowledge level
        4. Return results with all conversations
        
        Note: Conversations stop when:
        - System provides SQL (for answerable/solvable questions)
        - System provides feedback (for unsolvable questions)
        - Max steps + 1 is reached (extra step for final feedback)
        
        Evaluation flags (recognition, classification, solved, explained) are not set here.
        Use evaluators after running the benchmark to compute these metrics.
        """
        # Initialize conversations for each question and knowledge level
        conversations: list[Conversation] = []
        question_conversation_mapping: dict[int, list[int]] = {x: [] for x in range(len(questions))}
        for idx, question in enumerate(questions):
            conv_idx: int = idx * len(UserKnowledgeLevel) * len(CategoryUse)
            for knowledge_level in UserKnowledgeLevel:
                for category_use in CategoryUse:
                    conversations.append(Conversation(
                        question=question,
                        interactions=[],
                        user_knowledge_level=knowledge_level,
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
        """
        Run all evaluators on the benchmark results.
        
        This method is called automatically at the end of run().
        It can also be called manually on saved results.
        
        The evaluators set the following flags on each conversation:
        - recognition: True if system identified answerable vs unanswerable correctly
        - classification: True if system identified the exact category correctly
        - solved: True if predicted SQL matches ground truth SQL
        - explained: True if system's feedback matches expected feedback (for unsolvable questions)
        """
        for evaluator in self.evaluators:
            evaluator.evaluate(results.conversations)
