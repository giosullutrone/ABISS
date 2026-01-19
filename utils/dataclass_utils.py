from dataclasses import asdict
from enum import Enum


def generic_to_dict(obj) -> dict:
    return asdict(obj, dict_factory=lambda x: {
        k: (v.to_dict() if hasattr(v, "to_dict") else (v.value if isinstance(v, Enum) else v)) 
        for k, v in x
    })