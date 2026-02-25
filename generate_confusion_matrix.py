"""Generate classification confusion matrix from interaction results."""
import json
import csv
import os
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Config
RESULTS_DIR = Path("results/interaction")
TARGET_MODELS = [
    "Llama-3.3-70B-Instruct",
    "Qwen2.5-Coder-32B-Instruct",
    "Qwen2.5-32B-Instruct",
    "gemma-3-27b-it",
    "Mistral-Small-3.2-24B-Instruct-2506",
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
]

SHORT_NAMES = {
    ("Answerable", "With Evidence"): "Ans. w/ Ev.",
    ("Answerable", "Without Evidence"): "Ans. w/o Ev.",
    ("Conflicting Knowledge", None): "Confl. Know.",
    ("Improper Question", None): "Improper Q.",
    ("Lexical Vagueness", None): "Lex. Vague.",
    ("Missing External Knowledge", None): "Miss. Ext. Know.",
    ("Missing Schema Elements", "Missing Entities or Attributes"): "Miss. Sch. Ent.",
    ("Missing Schema Elements", "Missing Relationships"): "Miss. Sch. Rel.",
    ("Missing User Knowledge", None): "Miss. User Know.",
    ("Semantic Mapping Ambiguity", "Entity Ambiguity"): "Sem. Entity",
    ("Semantic Mapping Ambiguity", "Lexical Overlap"): "Sem. Lexical",
    ("Structure Ambiguity", "Attachment Ambiguity"): "Struct. Attach.",
    ("Structure Ambiguity", "Scope Ambiguity"): "Struct. Scope",
}

# Ordered list matching the paper's table order
CATEGORY_ORDER = [
    ("Answerable", "Without Evidence"),
    ("Answerable", "With Evidence"),
    ("Lexical Vagueness", None),
    ("Semantic Mapping Ambiguity", "Entity Ambiguity"),
    ("Semantic Mapping Ambiguity", "Lexical Overlap"),
    ("Structure Ambiguity", "Attachment Ambiguity"),
    ("Structure Ambiguity", "Scope Ambiguity"),
    ("Conflicting Knowledge", None),
    ("Missing User Knowledge", None),
    ("Improper Question", None),
    ("Missing External Knowledge", None),
    ("Missing Schema Elements", "Missing Entities or Attributes"),
    ("Missing Schema Elements", "Missing Relationships"),
]

DATASET_QUESTION_PATHS = {
    "bird_dev": "example_results/dev_generated_question_v16_merged.json",
    "spider_test": "example_results/spider_test_generated_question_v16_merged.json",
}

DATASET_DB_IDS = {
    "bird_dev": [
        "california_schools", "card_games", "codebase_community",
        "debit_card_specializing", "european_football_2", "financial",
        "formula_1", "student_club", "superhero", "thrombosis_prediction",
        "toxicology",
    ],
    "spider_test": [
        "company_1", "company_employee", "company_office", "hr_1",
        "customer_complaints", "customer_deliveries", "customers_and_orders",
        "customers_card_transactions", "department_store", "e_commerce",
        "store_product",
    ],
}


def get_balanced_counts(dataset):
    from utils.balancing import balance_questions, group_key_from_dict
    with open(DATASET_QUESTION_PATHS[dataset]) as f:
        raw = json.load(f)
    db_ids_set = set(DATASET_DB_IDS[dataset])
    qs = [q for q in raw if q["db_id"] in db_ids_set]
    balanced = balance_questions(qs, group_key_from_dict, seed=42)
    return Counter(
        (q["question"], q["db_id"], q["category"]["name"], q["category"].get("subname"))
        for q in balanced
    )


def build_confusion_matrices(balanced_cache):
    """Build confusion matrices from interaction results."""
    # confusion[(dataset, gt_key)][pred_key] = count
    confusion = defaultdict(Counter)
    active_counters = {}

    for filepath in sorted(RESULTS_DIR.glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)
        model = data["agent_name"]
        dataset = data["dataset_name"]
        if model not in TARGET_MODELS:
            continue

        for conv in data["conversations"]:
            if conv["category_use"] != "predicted":
                continue

            pred_cat = conv.get("predicted_category")
            if pred_cat is None:
                continue

            # Apply balanced filtering
            counter_key = (model, dataset, "predicted")
            if counter_key not in active_counters:
                active_counters[counter_key] = balanced_cache[dataset].copy()

            q_key = (
                conv["question"]["question"],
                conv["question"]["db_id"],
                conv["question"]["category"]["name"],
                conv["question"]["category"].get("subname"),
            )
            if active_counters[counter_key][q_key] <= 0:
                continue
            active_counters[counter_key][q_key] -= 1

            gt_cat = conv["question"]["category"]
            gt_key = (gt_cat["name"], gt_cat.get("subname"))
            pred_key = (pred_cat["name"], pred_cat.get("subname"))

            confusion[(dataset, gt_key)][pred_key] += 1

    return confusion


def plot_confusion_matrix(confusion, dataset, output_dir):
    """Plot and save confusion matrix heatmap."""
    ds_name = "BIRD" if "bird" in dataset else "Spider"

    # Build matrix
    n = len(CATEGORY_ORDER)
    matrix = np.zeros((n, n))
    labels = [SHORT_NAMES[c] for c in CATEGORY_ORDER]

    for i, gt_cat in enumerate(CATEGORY_ORDER):
        row_total = sum(confusion[(dataset, gt_cat)].values())
        if row_total > 0:
            for j, pred_cat in enumerate(CATEGORY_ORDER):
                count = confusion[(dataset, gt_cat)].get(pred_cat, 0)
                matrix[i, j] = 100 * count / row_total

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a mask for zero values to keep them white
    mask = matrix == 0

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        mask=mask,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": "Percentage (%)"},
        annot_kws={"size": 7},
    )

    # Add group separators
    # Groups: Answerable (0-1), Ambiguous (2-8), Unanswerable (9-12)
    for pos in [2, 9]:
        ax.axhline(y=pos, color="black", linewidth=2)
        ax.axvline(x=pos, color="black", linewidth=2)

    ax.set_xlabel("Predicted Subcategory", fontsize=12)
    ax.set_ylabel("Ground Truth Subcategory", fontsize=12)
    ax.set_title(f"Classification Confusion Matrix ({ds_name})", fontsize=14)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    output_path = output_dir / f"confusion_matrix_{ds_name.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")

    return matrix, labels


def save_confusion_csv(matrix, labels, dataset, output_dir):
    """Save confusion matrix as CSV."""
    ds_name = "BIRD" if "bird" in dataset else "Spider"
    csv_path = output_dir / f"confusion_matrix_{ds_name.lower()}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth"] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [f"{v:.1f}" for v in matrix[i]])
    print(f"Saved {csv_path}")


def main():
    output_dir = Path("charts/results_v16_balanced_group")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading balanced question counts...")
    balanced_cache = {}
    for ds in ["bird_dev", "spider_test"]:
        balanced_cache[ds] = get_balanced_counts(ds)
        total = sum(balanced_cache[ds].values())
        print(f"  {ds}: {total} balanced questions")

    print("\nBuilding confusion matrices from interaction results...")
    confusion = build_confusion_matrices(balanced_cache)

    for dataset in ["bird_dev", "spider_test"]:
        matrix, labels = plot_confusion_matrix(confusion, dataset, output_dir)
        save_confusion_csv(matrix, labels, dataset, output_dir)


if __name__ == "__main__":
    main()
