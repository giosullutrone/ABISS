from dataset_dataclasses.question import QuestionUnanswerable
from categories.category import Category


class CategoryMapper:
    def __init__(self, categories: list[type[Category]]) -> None:
        self.category_map: dict[tuple[str, str | None], type[Category]] = {(category.get_name(), category.get_subname()): category for category in categories}

    def get_category_from_question(self, question: QuestionUnanswerable) -> type[Category]:
        key = (question.category, question.subcategory)
        category = self.category_map.get(key)
        if category is None:
            raise ValueError(f"Category with name '{question.category}' and subname '{question.subcategory}' not found.")
        return category