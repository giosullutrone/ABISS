import json
import traceback
from typing import Any
from json_repair import repair_json
from pydantic import BaseModel


def clean_json_string(json_str: str) -> str:
    return json_str.replace('\n', ' ')\
        .replace('\r', ' ')\
        .replace('\t', ' ')\
        .replace('`', "'")\
        .replace('“', '"')\
        .replace('”', '"')\
        .replace("‘", "'")\
        .replace("’", "'")\
        .replace("\\.", ".")\
        .replace("\\_", "_")

def extract_last_json_object(text: str, constraint: type[BaseModel]) -> BaseModel | None:
    """
    Finds the last outermost JSON object in a string and validates it against a Pydantic model.
    Handles malformed JSON by attempting repair using json_repair.
    Also normalizes keys by converting to lowercase and replacing '-' and ' ' with '_'.
    """
    def normalize_key(key: str) -> str:
        """Normalize a key by converting to lowercase and replacing '-' and ' ' with '_'."""
        return key.lower().replace('-', '_').replace(' ', '_')
    
    def normalize_data_keys(data: dict, constraint: type[BaseModel]) -> dict:
        """
        Normalize data keys to match BaseModel field names.
        Both sides are normalized for comparison, but the ORIGINAL BaseModel field names
        are preserved in the output (e.g., if BaseModel has 'SQL-Query', it stays 'SQL-Query').
        """
        # Create mapping: normalized_key -> original_BaseModel_field_name
        field_mapping = {}
        for field_name in constraint.model_fields.keys():
            normalized = normalize_key(field_name)
            # Store original field name exactly as defined in BaseModel
            field_mapping[normalized] = field_name
        
        # Map data keys to BaseModel field names via normalization
        normalized_data = {}
        for key, value in data.items():
            normalized_key = normalize_key(key)
            if normalized_key in field_mapping:
                # Use the ORIGINAL BaseModel field name (preserves case, hyphens, etc.)
                normalized_data[field_mapping[normalized_key]] = value
            else:
                # Keep the original key if no match found
                normalized_data[key] = value
        
        return normalized_data
    
    # We search from the end of the string to find the last '}'
    end_idx = text.rfind('}')
    if end_idx == -1:
        return None

    # Stack-based approach to find the matching opening brace '{'
    brace_count = 0
    start_idx = -1
    
    for i in range(end_idx, -1, -1):
        if text[i] == '}':
            brace_count += 1
        elif text[i] == '{':
            brace_count -= 1
            
        # When count hits 0, we've found the outermost matching brace
        if brace_count == 0:
            start_idx = i
            break

    if start_idx == -1:
        return None

    json_str = text[start_idx : end_idx + 1]
    json_str = clean_json_string(json_str)

    try:
        # Try to parse and validate the JSON without repairing
        data = json.loads(json_str)
        # Only normalize if data is a dict, not a list or other type
        if isinstance(data, dict):
            normalized_data = normalize_data_keys(data, constraint)
            return constraint.model_validate(normalized_data)
        else:
            return None
    except:
        try:
            # If first attempt fails, try to repair the JSON
            repaired = repair_json(json_str)
            data = json.loads(repaired)
            # Only normalize if data is a dict, not a list or other type
            if isinstance(data, dict):
                normalized_data = normalize_data_keys(data, constraint)
                return constraint.model_validate(normalized_data)
            else:
                return None
        except:
            print(f"Failed to repair/validate JSON")
            traceback.print_exc()
            return None

def reorder_by_prefix_similarity(
    data: list[list[dict[str, str]]], 
    *other_lists: list[Any]
) -> tuple[list[list[dict[str, str]]], list[int], tuple[list[Any], ...]]:
    """
    Reorders a list of nested dictionaries so that items with similar prefixes are grouped together.
    This improves VLLM offline inference performance by maximizing prefix cache hits.
    Also reorders any additional lists in the same way to maintain synchronization.
    
    Args:
        data: List of lists of dictionaries to reorder (used for sorting)
        *other_lists: Additional lists to reorder in the same way as data
        
    Returns:
        tuple containing:
        - reordered_data: List of items sorted by prefix similarity
        - original_indices: Mapping from new position to original position
                           (use this with restore_original_order)
        - reordered_other_lists: Tuple of other lists reordered to match data
    
    Example:
        >>> data = [[{"key": "Hello world"}], [{"key": "Hi there"}]]
        >>> labels = ["label1", "label2"]
        >>> reordered_data, mapping, (reordered_labels,) = reorder_by_prefix_similarity(data, labels)
        >>> # All lists are reordered consistently based on data's string representation
    """
    if not data:
        return [], [], tuple([] for _ in other_lists)
    
    # Convert to strings for comparison
    string_representations = convert_nested_dicts_to_strings(data)
    
    # Create list of (index, data_item, string_rep) tuples
    indexed_data = list(enumerate(zip(data, string_representations)))
    
    # Sort by the string representations - this groups similar prefixes together
    # Python's sort is stable and lexicographic ordering naturally groups prefixes
    indexed_data.sort(key=lambda x: x[1][1])
    
    # Extract the sorted data items and their original indices
    reordered_data = [item for _, (item, _) in indexed_data]
    original_indices = [i for i, _ in indexed_data]
    
    # Reorder other lists using the same index order
    reordered_other_lists = tuple(
        [other_list[i] for i, _ in indexed_data]
        for other_list in other_lists
    )
    
    return reordered_data, original_indices, reordered_other_lists


def restore_original_order(reordered_list: list[Any], original_indices: list[int]) -> list[Any]:
    """
    Restores a list to its original order using the mapping from reorder_by_prefix_similarity.
    
    Args:
        reordered_list: The reordered list (e.g., results from VLLM inference)
        original_indices: The mapping returned by reorder_by_prefix_similarity
        
    Returns:
        List restored to original order
        
    Example:
        >>> data = [[{"key": "Hello world"}], [{"key": "Hi there"}]]
        >>> reordered, mapping = reorder_by_prefix_similarity(data)
        >>> # Process reordered data with VLLM
        >>> results = process_with_vllm(reordered)
        >>> # Restore original order
        >>> original_order_results = restore_original_order(results, mapping)
    """
    if not reordered_list:
        return []
    
    # Create a list to hold results in original order
    original_order: list[Any] = [None] * len(reordered_list)
    
    # Place each item back to its original position
    for new_idx, original_idx in enumerate(original_indices):
        original_order[original_idx] = reordered_list[new_idx]
    return original_order


def convert_nested_dicts_to_strings(data: list[list[dict[str, str]]]) -> list[str]:
    """
    Converts a nested list of dictionaries to a flat list of strings.
    
    Args:
        data: List of lists of dictionaries with string keys and values
        
    Returns:
        List of strings, one for each inner list converted to its string representation
        
    Example:
        >>> data = [[{"key": "value"}], [{"a": "1"}, {"b": "2"}]]
        >>> result = convert_nested_dicts_to_strings(data)
        >>> # result: ['[{"key": "value"}]', '[{"a": "1"}, {"b": "2"}]']
    """
    return [json.dumps(item) for item in data]
