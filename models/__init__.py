import json
from pydantic import BaseModel, ValidationError


def extract_last_json_object(text: str, constraint: type[BaseModel]) -> BaseModel | None:
    """
    Finds the last outermost JSON object in a string and validates it against a Pydantic model.
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
        # Clean up potential markdown artifacts if necessary
        data = json.loads(json_str)
        return constraint.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None