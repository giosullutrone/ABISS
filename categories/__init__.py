from categories.category import Category
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

def get_category_by_name(name: str, subname: str | None = None) -> Category | None:
    for category in get_all_categories():
        if category.get_name() == name and (subname is None or category.get_subname() == subname):
            return category
    return None

__all__ = [
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
