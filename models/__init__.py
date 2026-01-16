import json
import traceback
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
        normalized_data = normalize_data_keys(data, constraint)
        return constraint.model_validate(normalized_data)
    except:
        try:
            # If first attempt fails, try to repair the JSON
            repaired = repair_json(json_str)
            data = json.loads(repaired)
            normalized_data = normalize_data_keys(data, constraint)
            return constraint.model_validate(normalized_data)
        except:
            print(f"Failed to repair/validate JSON")
            traceback.print_exc()
            return None
