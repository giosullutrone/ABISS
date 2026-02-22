import json
import re
import traceback
from typing import Any
from json_repair import repair_json
from pydantic import BaseModel


def remove_json_comments(json_str: str) -> str:
    """
    Removes // single-line comments and /* */ multi-line comments from a JSON string.
    This is useful for cleaning LLM-generated JSON that may include comments.
    """
    # Remove /* */ multi-line comments
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    # Remove // single-line comments
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    return json_str


def clean_json_string(json_str: str) -> str:
    # Convert Python triple-quoted literals into JSON string literals
    def _triple_replacer(match: re.Match) -> str:
        inner = match.group(2)
        return json.dumps(inner)

    json_str = re.sub(r'(?P<quote>"""|\'\'\')(.*?)(?P=quote)', _triple_replacer, json_str, flags=re.DOTALL)

    return json_str.replace('\n', ' ')\
        .replace('\r', ' ')\
        .replace('\t', ' ')\
        .replace('“', '"')\
        .replace('”', '"')\
        .replace("‘", "'")\
        .replace("’", "'")\
        .replace("\\.", ".")\
        .replace("\\\\", "\\")\
        .replace("\\_", "_")\
        .replace("```json", "")\
        .replace("```", "")

def extract_last_json_object(text: str, constraint: type[BaseModel]) -> BaseModel | None:
    """
    Finds the last outermost JSON object in a string and validates it against a Pydantic model.
    Handles malformed JSON by attempting repair using json_repair.
    Also normalizes keys by converting to lowercase and replacing '-' and ' ' with '_'.
    """
    def normalize_key(key: str) -> str:
        """Normalize a key by converting to lowercase and replacing '-' and ' ' with '_', and handling escaped underscores."""
        return key.lower().replace('-', '_').replace(' ', '_').replace('\\_', '_')
    
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
    
    # Find the last '}' and then locate its matching opening '{' by scanning
    # backward.  This ensures we pick the *last* outermost JSON object rather
    # than the first '{' that happens to pair with the last '}'.
    end_idx = text.rfind('}')
    if end_idx == -1:
        return None

    start_idx = -1

    # Walk backward from end_idx to find the matching '{'.
    for i in range(end_idx, -1, -1):
        if text[i] != '{':
            continue

        brace_count = 0
        in_string = False
        escape = False
        matched = False

        j = i
        while j <= end_idx:
            ch = text[j]

            if escape:
                escape = False
                j += 1
                continue

            if ch == '\\' and in_string:
                escape = True
                j += 1
                continue

            if ch == '"':
                in_string = not in_string
                j += 1
                continue

            if not in_string:
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0 and j == end_idx:
                        matched = True
                        break

            j += 1

        if matched:
            start_idx = i
            break

    if start_idx == -1:
        return None

    json_str = text[start_idx:end_idx + 1]

    json_str = remove_json_comments(json_str)
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
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
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
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e2:
            print(f"Failed to repair/validate JSON")
            traceback.print_exc()
            return None

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
