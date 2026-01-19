from categories.category import Category
from categories.answerable import AnswerableCategory
from categories.conflicting_knowledge import ConflictingKnowledgeCategory
from categories.improper_question import ImproperQuestionCategory
from categories.lexical_vagueness import LexicalVaguenessCategory
from categories.missing_external_knowledge import MissingExternalKnowledgeCategory
from categories.missing_schema_entities import MissingSchemaEntitiesCategory
from categories.missing_schema_relationships import MissingSchemaRelationshipsCategory
from categories.missing_user_knowledge import MissingUserKnowledgeCategory
from categories.semantic_mapping_entity_ambiguity import SemanticMappingEntityAmbiguityCategory
from categories.semantic_mapping_lexical_overlap import SemanticMappingLexicalOverlapCategory
from categories.structure_ambiguity_attachment import StructureAmbiguityAttachmentCategory
from categories.structure_ambiguity_scope import StructureAmbiguityScopeCategory

def get_all_categories() -> list[Category]:
    return [
        AnswerableCategory(),
        ConflictingKnowledgeCategory(),
        ImproperQuestionCategory(),
        LexicalVaguenessCategory(),
        MissingExternalKnowledgeCategory(),
        MissingSchemaEntitiesCategory(),
        MissingSchemaRelationshipsCategory(),
        MissingUserKnowledgeCategory(),
        SemanticMappingEntityAmbiguityCategory(),
        SemanticMappingLexicalOverlapCategory(),
        StructureAmbiguityAttachmentCategory(),
        StructureAmbiguityScopeCategory(),
    ]

def get_category_by_name_and_subname(name: str, subname: str | None = None, fuzzy: bool = False) -> Category | None:
    if fuzzy:
        name = name.lower().replace(" ", "")
        for category in get_all_categories():
            cat_name = category.get_name().lower().replace(" ", "")
            cat_subname = category.get_subname()
            cat_subname_clean = cat_subname.lower().replace(" ", "") if cat_subname is not None else None
            if cat_name == name and (subname is None or (cat_subname_clean == subname.lower().replace(" ", "") if cat_subname_clean is not None else False)):
                return category
        return AnswerableCategory()  # Default to Answerable if no match found
    
    for category in get_all_categories():
        if category.get_name() == name and (subname is None or category.get_subname() == subname):
            return category
    return None

def get_category_by_class_name(name: str) -> Category | None:
    for category in get_all_categories():
        if category.__class__.__name__ == name:
            return category
    return None

__all__ = [
    "AnswerableCategory",
    "ConflictingKnowledgeCategory",
    "ImproperQuestionCategory",
    "LexicalVaguenessCategory",
    "MissingExternalKnowledgeCategory",
    "MissingSchemaEntitiesCategory",
    "MissingSchemaRelationshipsCategory",
    "MissingUserKnowledgeCategory",
    "SemanticMappingEntityAmbiguityCategory",
    "SemanticMappingLexicalOverlapCategory",
    "StructureAmbiguityAttachmentCategory",
    "StructureAmbiguityScopeCategory",
    "get_all_categories"
]
