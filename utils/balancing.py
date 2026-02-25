"""Balanced sampling utility for benchmark questions.

Downsamples a set of questions so that every group (defined by a key
function) has equal count. Used by both do_interaction.py (on dataclass
objects) and analyze_interactions.py (on raw dicts).
"""
import random
from collections import defaultdict


def balance_questions(items, key_fn, seed=42):
    """Downsample items so every group (defined by key_fn) has equal count.

    Args:
        items: list of questions (dataclass objects or dicts).
        key_fn: callable that extracts the grouping key from an item.
        seed: random seed for reproducibility.

    Returns:
        list of items after balanced sampling, preserving within-group
        ordering for groups that are already at the minimum count.
    """
    groups = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)

    min_count = min(len(v) for v in groups.values())

    rng = random.Random(seed)
    balanced = []
    for key in sorted(groups.keys()):
        group = groups[key]
        if len(group) > min_count:
            balanced.extend(rng.sample(group, min_count))
        else:
            balanced.extend(group)

    return balanced


# -- Key functions for Question dataclass objects (do_interaction.py) --------

def category_key_from_dataclass(q):
    """Extract (name, subname) category key from a Question dataclass."""
    return (q.category.get_name(), q.category.get_subname())


def group_key_from_dataclass(q):
    """Extract group key (Answerable/Ambiguous/Unanswerable) from a Question dataclass."""
    if q.category.is_answerable():
        return "Answerable"
    elif q.category.is_solvable():
        return "Ambiguous"
    else:
        return "Unanswerable"


# -- Key functions for raw question dicts (analyze_interactions.py) ----------

def category_key_from_dict(d):
    """Extract (name, subname) category key from a question dict."""
    cat = d["category"]
    return (cat["name"], cat.get("subname"))


def group_key_from_dict(d):
    """Extract group key (Answerable/Ambiguous/Unanswerable) from a question dict."""
    cat = d["category"]
    if cat["answerable"]:
        return "Answerable"
    elif cat["solvable"]:
        return "Ambiguous"
    else:
        return "Unanswerable"
