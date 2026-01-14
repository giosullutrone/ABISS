from pydantic import BaseModel


def model_field_descriptions(model: type[BaseModel]) -> str:
    lines = ["{"]

    for name, field in model.model_fields.items():
        assert field.description is not None, f"Field {name} is missing description."
        desc = field.description
        lines.append(f'    "{name}": "{desc}",')

    lines.append("}")
    return "\n".join(lines)
