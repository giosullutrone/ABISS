from pydantic import Field, BaseModel
from typing import Annotated
from categories.category import Category
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_dataclasses.question import Question, QuestionStyle, QuestionDifficulty


class LexicalVaguenessCategory(Category):
    class LexicalVaguenessOutput(BaseModel):
        question: Annotated[str, Field(description="A natural language question containing a vague term whose meaning lacks a precise or objective boundary — such as temporal expressions ('recent', 'old'), quantitative adjectives ('many', 'few', 'high', 'low'), or evaluative terms ('good', 'popular', 'expensive'). The ambiguity is about the THRESHOLD or CUTOFF for a gradable term, NOT about sentence structure, NOT about which column a word maps to, NOT about user identity, and NOT about conflicting evidence definitions.")]
        hidden_knowledge_first_interpretation: Annotated[str, Field(description="A statement providing a concrete threshold or cutoff for the vague term. It should replace the vague expression with a specific, objective criterion. For example: 'The term recent means within the last month.'")]
        hidden_knowledge_second_interpretation: Annotated[str, Field(description="A statement providing an alternative concrete threshold for the same vague term. It should offer a meaningfully different cutoff that is equally plausible. For example: 'The term recent means within the last academic year.'")]
        sql_first_interpretation: Annotated[str, Field(description="A valid, executable SQL query using the first concrete threshold. The two SQL variants must differ in the WHERE clause threshold or HAVING condition value, reflecting different cutoff interpretations of the same vague term. The query must correctly answer the question under this interpretation. Use only the first threshold — do not include conditions from the alternative threshold.")]
        sql_second_interpretation: Annotated[str, Field(description="A valid, executable SQL query using the second concrete threshold. The two SQL variants must differ in the WHERE clause threshold or HAVING condition value, reflecting different cutoff interpretations of the same vague term. The query must correctly answer the question under this interpretation. Use only the second threshold — do not include conditions from the alternative threshold.")]

    @staticmethod
    def get_name() -> str:
        return "Lexical Vagueness"

    @staticmethod
    def get_subname() -> str | None:
        return None

    @staticmethod
    def get_definition() -> str:
        return (
            "Lexical Vagueness arises when a question contains terms whose meaning lacks a precise or objective boundary, "
            "leading to indeterminate selection criteria during query generation. "
            "These are gradable terms — temporal ('recent', 'old'), quantitative ('many', 'few', 'high', 'low'), "
            "or evaluative ('good', 'popular', 'expensive') — that require a subjective threshold or cutoff to become concrete. "
            "The ambiguity is purely about where to draw the line for an inherently imprecise term, "
            "not about which schema element to use or how the sentence is structured. "
            "Important: This is NOT about how a modifier attaches in a conjunction (Attachment Ambiguity), "
            "NOT about quantifier scope (Scope Ambiguity), "
            "NOT about which database table or column a word maps to (Entity Ambiguity / Lexical Overlap), "
            "NOT about user-specific references like 'my' or 'our' (Missing User Knowledge), "
            "and NOT about conflicting evidence definitions from the knowledge base (Conflicting Knowledge)."
        )

    @staticmethod
    def get_examples() -> list[str] | None:
        return [
            "List recent courses. (Lexical vagueness: 'recent' could mean within the last month or within the last academic year — no objective temporal boundary)",
            "Show employees with high salaries. (Lexical vagueness: 'high' could mean above $100,000 or above the company median — no objective quantitative threshold)",
            "Find popular products. (Lexical vagueness: 'popular' could mean top 10 by sales volume or those with more than 100 reviews — no objective evaluative criterion)",
            "Display old buildings on campus. (Lexical vagueness: 'old' could mean built before 1950 or built more than 50 years ago — no objective temporal boundary)"
        ]

    @staticmethod
    def is_answerable() -> bool:
        return False

    @staticmethod
    def is_solvable() -> bool:
        return True

    @staticmethod
    def get_output() -> type[BaseModel]:
        return LexicalVaguenessCategory.LexicalVaguenessOutput

    @staticmethod
    def get_question(db_id: str, output: BaseModel, question_style: "QuestionStyle", question_difficulty: "QuestionDifficulty") -> list["Question"]:
        from dataset_dataclasses.question import QuestionUnanswerable
        assert isinstance(output, LexicalVaguenessCategory.LexicalVaguenessOutput)
        return [QuestionUnanswerable(
            db_id=db_id,
            category=LexicalVaguenessCategory(),
            question=output.question,
            evidence=None,
            sql=sql,
            hidden_knowledge=hk,
            is_solvable=LexicalVaguenessCategory.is_solvable(),
            question_style=question_style,
            question_difficulty=question_difficulty
        ) for sql, hk in [
            (output.sql_first_interpretation, output.hidden_knowledge_first_interpretation),
            (output.sql_second_interpretation, output.hidden_knowledge_second_interpretation)
        ]]
