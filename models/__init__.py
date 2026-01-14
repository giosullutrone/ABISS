import json
from json_repair import repair_json
from pydantic import BaseModel, ValidationError


def extract_last_json_object(text: str, constraint: type[BaseModel]) -> BaseModel | None:
    """
    Finds the last outermost JSON object in a string and validates it against a Pydantic model.
    Handles malformed JSON by attempting repair using json_repair.
    """
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

    try:
        data = json.loads(json_str)
        return constraint.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        # If first attempt fails, try to repair the JSON
        if isinstance(e, json.JSONDecodeError):
            try:
                repaired = repair_json(json_str)
                data = json.loads(repaired)
                return constraint.model_validate(data)
            except (json.JSONDecodeError, ValidationError, Exception):
                return None
        return None
