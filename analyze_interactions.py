"""Analyze interaction results and generate LaTeX tables + CSV summaries for paper_v3.tex."""
import argparse
import json
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Target models (near-complete data)
TARGET_MODELS = [
    "Llama-3.3-70B-Instruct",
    "Qwen2.5-Coder-32B-Instruct",
    "Qwen2.5-32B-Instruct",
    "gemma-3-27b-it",
    "Mistral-Small-3.2-24B-Instruct-2506",
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
]

MODEL_DISPLAY_NAMES = {
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B",
    "Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "gemma-3-27b-it": "Gemma-3-27B",
    "Mistral-Small-3.2-24B-Instruct-2506": "Mistral-Small-24B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
}

MODEL_SIZES = {
    "Llama-3.3-70B-Instruct": "70B",
    "Qwen2.5-Coder-32B-Instruct": "32B",
    "Qwen2.5-32B-Instruct": "32B",
    "gemma-3-27b-it": "27B",
    "Mistral-Small-3.2-24B-Instruct-2506": "24B",
    "Llama-3.1-8B-Instruct": "8B",
    "Qwen2.5-7B-Instruct": "7B",
}

DATASET_NAMES = {
    "bird_dev": "BIRD",
    "spider_test": "Spider",
}

CATEGORY_SHORT_NAMES = {
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

CATEGORY_LATEX_NAMES = {
    ("Answerable", "With Evidence"): "Answerable w/ Evidence",
    ("Answerable", "Without Evidence"): "Answerable w/o Evidence",
    ("Conflicting Knowledge", None): "Conflicting Knowledge",
    ("Improper Question", None): "Improper Question",
    ("Lexical Vagueness", None): "Lexical Vagueness",
    ("Missing External Knowledge", None): "Missing External Knowledge",
    ("Missing Schema Elements", "Missing Entities or Attributes"): "Missing Schema Entities",
    ("Missing Schema Elements", "Missing Relationships"): "Missing Schema Relationships",
    ("Missing User Knowledge", None): "Missing User Knowledge",
    ("Semantic Mapping Ambiguity", "Entity Ambiguity"): "Semantic Mapping (Entity)",
    ("Semantic Mapping Ambiguity", "Lexical Overlap"): "Semantic Mapping (Lexical)",
    ("Structure Ambiguity", "Attachment Ambiguity"): "Structural (Attachment)",
    ("Structure Ambiguity", "Scope Ambiguity"): "Structural (Scope)",
}

# Category ordering: Answerable, Ambiguous, Unanswerable
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

# Dataset configuration for balanced mode (question paths and db_ids used during benchmark)
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


def get_category_key(conv):
    """Extract (name, subname) tuple from conversation."""
    q = conv["question"]
    cat = q["category"]
    return (cat["name"], cat.get("subname"))


def get_category_group(conv):
    """Get the category group: Answerable, Ambiguous, or Unanswerable."""
    cat = conv["question"]["category"]
    if cat.get("answerable", False):
        return "Answerable"
    elif cat.get("solvable", False):
        return "Ambiguous"
    else:
        return "Unanswerable"


def compute_rate(values):
    """Compute rate as True / (True + False), ignoring None."""
    true_count = sum(1 for v in values if v is True)
    false_count = sum(1 for v in values if v is False)
    total = true_count + false_count
    if total == 0:
        return None
    return true_count / total


def load_all_results(results_dir):
    """Load all result files for target models, return list of (model, dataset, conversations)."""
    results_dir = Path(results_dir)
    all_data = []

    for filepath in sorted(results_dir.glob("*.json")):
        with open(filepath, "r") as f:
            data = json.load(f)

        model = data["agent_name"]
        dataset = data["dataset_name"]

        if model not in TARGET_MODELS:
            continue

        all_data.append((model, dataset, data["conversations"]))
        print(f"  Loaded {filepath.name}: {len(data['conversations'])} conversations")

    return all_data


def aggregate_metrics(all_data):
    """Aggregate all metrics into a structured dictionary.

    Returns a dict keyed by (model, dataset, category_use) with values containing
    lists of metric values for recognition, classification, solved, explained,
    plus interaction data.
    """
    metrics = defaultdict(lambda: {
        "recognition": [],
        "classification": [],
        "solved": [],
        "explained": [],
        "n_interactions": [],
        "relevancy_labels": [],
        "categories": [],
        "category_groups": [],
        "difficulties": [],
        "has_clarification": [],
    })

    for model, dataset, conversations in all_data:
        for conv in conversations:
            cat_use = conv["category_use"]
            key = (model, dataset, cat_use)
            cat_key = get_category_key(conv)
            cat_group = get_category_group(conv)
            difficulty = conv["question"]["question_difficulty"]

            metrics[key]["recognition"].append(conv["recognition"])
            metrics[key]["classification"].append(conv["classification"])
            metrics[key]["solved"].append(conv["solved"])
            metrics[key]["explained"].append(conv["explained"])
            metrics[key]["categories"].append(cat_key)
            metrics[key]["category_groups"].append(cat_group)
            metrics[key]["difficulties"].append(difficulty)

            interactions = conv.get("interactions", [])
            metrics[key]["n_interactions"].append(len(interactions))

            asked_question = False
            for inter in interactions:
                sr = inter["system_response"]
                if sr.get("system_question") is not None:
                    asked_question = True
                rel = inter.get("relevance")
                if rel is not None:
                    metrics[key]["relevancy_labels"].append(rel)
            metrics[key]["has_clarification"].append(asked_question)

    return metrics


def print_overall_results_table(metrics):
    """Print Table 1: Overall results per model (PREDICTED mode)."""
    print("\n" + "=" * 80)
    print("TABLE 1: OVERALL RESULTS (PREDICTED mode)")
    print("=" * 80)

    print("\n% LaTeX Table: Overall Results")
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Overall evaluation results on BIRD and Spider benchmarks. "
          "All metrics are reported under the \\textit{Predicted} category mode, "
          "where the system must first classify each question before responding. "
          "Rec.~= Recognition accuracy, Cls.~= Classification accuracy, "
          "EX~= Execution accuracy, FB~= Feedback quality.}")
    print("\\label{tab:overall_results}")
    print("\\small")
    print("\\begin{tabular}{@{}llrrrrrrrr@{}}")
    print("\\toprule")
    print("& & \\multicolumn{4}{c}{\\textbf{BIRD}} & \\multicolumn{4}{c}{\\textbf{Spider}} \\\\")
    print("\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}")
    print("\\textbf{Model} & \\textbf{Size} & \\textbf{Rec.} & \\textbf{Cls.} & \\textbf{EX} & \\textbf{FB} & \\textbf{Rec.} & \\textbf{Cls.} & \\textbf{EX} & \\textbf{FB} \\\\")
    print("\\midrule")

    for model in TARGET_MODELS:
        display = MODEL_DISPLAY_NAMES[model]
        size = MODEL_SIZES[model]
        row = f"{display} & {size}"

        for dataset in ["bird_dev", "spider_test"]:
            key = (model, dataset, "predicted")
            if key not in metrics:
                row += " & -- & -- & -- & --"
                continue

            m = metrics[key]
            rec = compute_rate(m["recognition"])
            cls_ = compute_rate(m["classification"])
            ex = compute_rate(m["solved"])
            fb = compute_rate(m["explained"])

            for val in [rec, cls_, ex, fb]:
                if val is None:
                    row += " & --"
                else:
                    row += f" & {val * 100:.1f}"

        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")

    # Also print plain text summary
    print("\n--- Plain text summary ---")
    for model in TARGET_MODELS:
        for dataset in ["bird_dev", "spider_test"]:
            key = (model, dataset, "predicted")
            if key not in metrics:
                continue
            m = metrics[key]
            n = len(m["recognition"])
            rec = compute_rate(m["recognition"])
            cls_ = compute_rate(m["classification"])
            ex = compute_rate(m["solved"])
            fb = compute_rate(m["explained"])
            print(f"  {MODEL_DISPLAY_NAMES[model]:20s} {DATASET_NAMES[dataset]:8s}: "
                  f"n={n:5d}  Rec={rec*100 if rec else 0:.1f}%  "
                  f"Cls={cls_*100 if cls_ else 0:.1f}%  "
                  f"EX={ex*100 if ex else 0:.1f}%  "
                  f"FB={fb*100 if fb else 0:.1f}%")


def print_ablation_table(metrics):
    """Print Table 2: Category knowledge ablation."""
    print("\n" + "=" * 80)
    print("TABLE 2: CATEGORY KNOWLEDGE ABLATION")
    print("=" * 80)

    # Aggregate across all models per (dataset, category_use)
    ablation = defaultdict(lambda: {"solved": [], "explained": [], "recognition": [], "classification": []})

    for (model, dataset, cat_use), m in metrics.items():
        key = (dataset, cat_use)
        ablation[key]["solved"].extend(m["solved"])
        ablation[key]["explained"].extend(m["explained"])
        ablation[key]["recognition"].extend(m["recognition"])
        ablation[key]["classification"].extend(m["classification"])

    print("\n% LaTeX Table: Category Knowledge Ablation")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Impact of category knowledge on benchmark performance, "
          "averaged across all seven models. Ground Truth provides the oracle category; "
          "Predicted requires the system to classify first; No Category provides no "
          "category information. Recognition and Classification are only applicable "
          "under the Predicted mode.}")
    print("\\label{tab:ablation}")
    print("\\small")
    print("\\begin{tabular}{@{}llrrrr@{}}")
    print("\\toprule")
    print("& \\textbf{Mode} & \\textbf{Rec.} & \\textbf{Cls.} & \\textbf{EX} & \\textbf{FB} \\\\")
    print("\\midrule")

    mode_labels = {
        "ground_truth": "Ground Truth",
        "predicted": "Predicted",
        "no_category": "No Category",
    }

    for dataset in ["bird_dev", "spider_test"]:
        ds_label = DATASET_NAMES[dataset]
        first = True
        for cat_use in ["ground_truth", "predicted", "no_category"]:
            key = (dataset, cat_use)
            if key not in ablation:
                continue

            a = ablation[key]
            rec = compute_rate(a["recognition"])
            cls_ = compute_rate(a["classification"])
            ex = compute_rate(a["solved"])
            fb = compute_rate(a["explained"])

            prefix = f"\\multirow{{3}}{{*}}{{{ds_label}}}" if first else ""
            first = False

            rec_str = f"{rec * 100:.1f}" if rec is not None else "--"
            cls_str = f"{cls_ * 100:.1f}" if cls_ is not None else "--"
            ex_str = f"{ex * 100:.1f}" if ex is not None else "--"
            fb_str = f"{fb * 100:.1f}" if fb is not None else "--"

            print(f"{prefix} & {mode_labels[cat_use]} & {rec_str} & {cls_str} & {ex_str} & {fb_str} \\\\")

        if dataset == "bird_dev":
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Plain text
    print("\n--- Plain text ablation ---")
    for dataset in ["bird_dev", "spider_test"]:
        for cat_use in ["ground_truth", "predicted", "no_category"]:
            key = (dataset, cat_use)
            if key not in ablation:
                continue
            a = ablation[key]
            ex = compute_rate(a["solved"])
            fb = compute_rate(a["explained"])
            rec = compute_rate(a["recognition"])
            cls_ = compute_rate(a["classification"])
            print(f"  {DATASET_NAMES[dataset]:8s} {cat_use:15s}: "
                  f"Rec={rec*100 if rec else 0:.1f}%  "
                  f"Cls={cls_*100 if cls_ else 0:.1f}%  "
                  f"EX={ex*100 if ex else 0:.1f}%  "
                  f"FB={fb*100 if fb else 0:.1f}%")


def print_per_category_table(metrics):
    """Print per-category breakdown (appendix table)."""
    print("\n" + "=" * 80)
    print("TABLE: PER-CATEGORY RESULTS (PREDICTED mode, averaged across models)")
    print("=" * 80)

    # Aggregate per (dataset, category)
    cat_metrics = defaultdict(lambda: {"solved": [], "explained": [], "recognition": [], "classification": []})

    for (model, dataset, cat_use), m in metrics.items():
        if cat_use != "predicted":
            continue
        for i, cat_key in enumerate(m["categories"]):
            key = (dataset, cat_key)
            cat_metrics[key]["solved"].append(m["solved"][i])
            cat_metrics[key]["explained"].append(m["explained"][i])
            cat_metrics[key]["recognition"].append(m["recognition"][i])
            cat_metrics[key]["classification"].append(m["classification"][i])

    print("\n% LaTeX Table: Per-Category Results")
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Per-category evaluation results under the Predicted mode, averaged across all seven models. "
          "Categories are grouped by type: Answerable (top), Ambiguous (middle), and Unanswerable (bottom). "
          "EX is reported for answerable and ambiguous categories; FB is reported for unanswerable categories.}")
    print("\\label{tab:per_category}")
    print("\\small")
    print("\\begin{tabular}{@{}llrrrrrrrr@{}}")
    print("\\toprule")
    print("& & \\multicolumn{4}{c}{\\textbf{BIRD}} & \\multicolumn{4}{c}{\\textbf{Spider}} \\\\")
    print("\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}")
    print("\\textbf{Group} & \\textbf{Category} & \\textbf{Rec.} & \\textbf{Cls.} & \\textbf{EX} & \\textbf{FB} & \\textbf{Rec.} & \\textbf{Cls.} & \\textbf{EX} & \\textbf{FB} \\\\")
    print("\\midrule")

    current_group = None
    for cat_key in CATEGORY_ORDER:
        cat_dict_answerable = cat_key[0] == "Answerable"
        cat_dict_solvable = cat_key[0] in ["Answerable", "Lexical Vagueness", "Semantic Mapping Ambiguity",
                                             "Structure Ambiguity", "Conflicting Knowledge", "Missing User Knowledge"]

        if cat_dict_answerable:
            group = "Answerable"
        elif cat_dict_solvable:
            group = "Ambiguous"
        else:
            group = "Unanswerable"

        group_prefix = ""
        if group != current_group:
            if current_group is not None:
                print("\\midrule")
            group_prefix = group
            current_group = group

        cat_label = CATEGORY_LATEX_NAMES.get(cat_key, f"{cat_key[0]} {cat_key[1] or ''}")
        row = f"{group_prefix} & {cat_label}"

        for dataset in ["bird_dev", "spider_test"]:
            key = (dataset, cat_key)
            if key not in cat_metrics:
                row += " & -- & -- & -- & --"
                continue

            cm = cat_metrics[key]
            rec = compute_rate(cm["recognition"])
            cls_ = compute_rate(cm["classification"])
            ex = compute_rate(cm["solved"])
            fb = compute_rate(cm["explained"])

            for val in [rec, cls_, ex, fb]:
                if val is None:
                    row += " & --"
                else:
                    row += f" & {val * 100:.1f}"

        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")


def print_per_difficulty_table(metrics):
    """Print per-difficulty breakdown."""
    print("\n" + "=" * 80)
    print("TABLE: PER-DIFFICULTY RESULTS (PREDICTED mode, averaged across models)")
    print("=" * 80)

    diff_metrics = defaultdict(lambda: {"solved": [], "explained": [], "recognition": [], "classification": []})

    for (model, dataset, cat_use), m in metrics.items():
        if cat_use != "predicted":
            continue
        for i, diff in enumerate(m["difficulties"]):
            key = (dataset, diff)
            diff_metrics[key]["solved"].append(m["solved"][i])
            diff_metrics[key]["explained"].append(m["explained"][i])
            diff_metrics[key]["recognition"].append(m["recognition"][i])
            diff_metrics[key]["classification"].append(m["classification"][i])

    print("\n% LaTeX Table: Per-Difficulty Results")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Evaluation results by SQL difficulty level under the Predicted mode, "
          "averaged across all seven models.}")
    print("\\label{tab:per_difficulty}")
    print("\\small")
    print("\\begin{tabular}{@{}llrrrr@{}}")
    print("\\toprule")
    print("& \\textbf{Difficulty} & \\textbf{Rec.} & \\textbf{Cls.} & \\textbf{EX} & \\textbf{FB} \\\\")
    print("\\midrule")

    diff_order = ["simple", "moderate", "complex", "highly_complex"]
    diff_labels = {"simple": "Simple", "moderate": "Moderate", "complex": "Complex", "highly_complex": "Highly Complex"}

    for dataset in ["bird_dev", "spider_test"]:
        ds_label = DATASET_NAMES[dataset]
        first = True
        for diff in diff_order:
            key = (dataset, diff)
            if key not in diff_metrics:
                continue
            dm = diff_metrics[key]
            rec = compute_rate(dm["recognition"])
            cls_ = compute_rate(dm["classification"])
            ex = compute_rate(dm["solved"])
            fb = compute_rate(dm["explained"])

            prefix = f"\\multirow{{4}}{{*}}{{{ds_label}}}" if first else ""
            first = False

            vals = []
            for v in [rec, cls_, ex, fb]:
                vals.append(f"{v * 100:.1f}" if v is not None else "--")

            print(f"{prefix} & {diff_labels[diff]} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\")

        if dataset == "bird_dev":
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def compute_ttr_rtf(all_data):
    """Compute TTR (Time-to-Relevant) and RTF (Relevant-to-Finish) metrics.

    TTR: 1-indexed turn number of the first RELEVANT clarification.
    RTF: number of turns after the first RELEVANT clarification until conversation end.
    Only conversations with at least one RELEVANT clarification are included.
    """
    print("\n" + "=" * 80)
    print("TTR / RTF ANALYSIS")
    print("=" * 80)

    # Collect per (model, dataset)
    ttr_rtf_data = defaultdict(lambda: {
        "ttr": [],          # turns to first relevant
        "rtf": [],          # turns from first relevant to end
        "total_convs": 0,   # total conversations with clarifications
        "no_relevant": 0,   # conversations with clarifications but no relevant one
    })

    for model, dataset, conversations in all_data:
        if model not in TARGET_MODELS:
            continue
        key = (model, dataset)

        for conv in conversations:
            interactions = conv.get("interactions", [])

            # Only consider conversations that had at least one clarification
            has_clarification = any(
                inter.get("system_response", {}).get("system_question") is not None
                for inter in interactions
            )
            if not has_clarification:
                continue

            ttr_rtf_data[key]["total_convs"] += 1

            # Find first relevant clarification
            first_relevant_idx = None
            for idx, inter in enumerate(interactions):
                if inter.get("relevance") == "Relevant":
                    first_relevant_idx = idx
                    break

            if first_relevant_idx is None:
                ttr_rtf_data[key]["no_relevant"] += 1
                continue

            ttr = first_relevant_idx + 1  # 1-indexed
            rtf = len(interactions) - 1 - first_relevant_idx  # turns after relevant to end

            ttr_rtf_data[key]["ttr"].append(ttr)
            ttr_rtf_data[key]["rtf"].append(rtf)

    # Print results per model
    print("\n--- Per-model TTR/RTF (conversations with at least one clarification) ---")
    print(f"  {'Model':<25s} {'Dataset':<8s} {'Convs':>7s} {'NoRel':>7s} {'NoRel%':>7s} "
          f"{'AvgTTR':>7s} {'AvgRTF':>7s} {'TTR=1%':>7s} {'RTF=0%':>7s} {'RTF=1%':>7s}")

    for model in TARGET_MODELS:
        for dataset in ["bird_dev", "spider_test"]:
            key = (model, dataset)
            if key not in ttr_rtf_data:
                continue

            d = ttr_rtf_data[key]
            total = d["total_convs"]
            no_rel = d["no_relevant"]
            no_rel_pct = no_rel / total * 100 if total else 0
            n_with_rel = len(d["ttr"])

            if n_with_rel > 0:
                avg_ttr = np.mean(d["ttr"])
                avg_rtf = np.mean(d["rtf"])
                ttr_1_pct = sum(1 for t in d["ttr"] if t == 1) / n_with_rel * 100
                rtf_0_pct = sum(1 for r in d["rtf"] if r == 0) / n_with_rel * 100
                rtf_1_pct = sum(1 for r in d["rtf"] if r <= 1) / n_with_rel * 100
            else:
                avg_ttr = avg_rtf = ttr_1_pct = rtf_0_pct = rtf_1_pct = 0

            print(f"  {MODEL_DISPLAY_NAMES[model]:<25s} {DATASET_NAMES[dataset]:<8s} "
                  f"{total:>7d} {no_rel:>7d} {no_rel_pct:>6.1f}% "
                  f"{avg_ttr:>7.2f} {avg_rtf:>7.2f} "
                  f"{ttr_1_pct:>6.1f}% {rtf_0_pct:>6.1f}% {rtf_1_pct:>6.1f}%")

    # Aggregate across all models
    print("\n--- Aggregated across all models ---")
    agg = defaultdict(lambda: {"ttr": [], "rtf": [], "total_convs": 0, "no_relevant": 0})
    for (model, dataset), d in ttr_rtf_data.items():
        agg[dataset]["ttr"].extend(d["ttr"])
        agg[dataset]["rtf"].extend(d["rtf"])
        agg[dataset]["total_convs"] += d["total_convs"]
        agg[dataset]["no_relevant"] += d["no_relevant"]

    for dataset in ["bird_dev", "spider_test"]:
        if dataset not in agg:
            continue
        d = agg[dataset]
        total = d["total_convs"]
        no_rel = d["no_relevant"]
        n_with_rel = len(d["ttr"])

        if n_with_rel > 0:
            avg_ttr = np.mean(d["ttr"])
            avg_rtf = np.mean(d["rtf"])
            ttr_1_pct = sum(1 for t in d["ttr"] if t == 1) / n_with_rel * 100
            rtf_0_pct = sum(1 for r in d["rtf"] if r == 0) / n_with_rel * 100
            rtf_1_pct = sum(1 for r in d["rtf"] if r <= 1) / n_with_rel * 100
        else:
            avg_ttr = avg_rtf = ttr_1_pct = rtf_0_pct = rtf_1_pct = 0

        print(f"  {DATASET_NAMES[dataset]:<8s}: total_with_clar={total}  no_relevant={no_rel} ({no_rel/total*100 if total else 0:.1f}%)  "
              f"avg_ttr={avg_ttr:.2f}  avg_rtf={avg_rtf:.2f}  "
              f"ttr=1: {ttr_1_pct:.1f}%  rtf=0: {rtf_0_pct:.1f}%  rtf<=1: {rtf_1_pct:.1f}%")


def print_interaction_analysis(metrics):
    """Print interaction analysis tables and statistics."""
    print("\n" + "=" * 80)
    print("INTERACTION ANALYSIS")
    print("=" * 80)

    # Aggregate interaction stats per (model, dataset) across all category_use modes
    interaction_stats = defaultdict(lambda: {
        "n_interactions": [],
        "has_clarification": [],
        "relevancy_labels": [],
        "category_groups": [],
    })

    for (model, dataset, cat_use), m in metrics.items():
        key = (model, dataset)
        interaction_stats[key]["n_interactions"].extend(m["n_interactions"])
        interaction_stats[key]["has_clarification"].extend(m["has_clarification"])
        interaction_stats[key]["relevancy_labels"].extend(m["relevancy_labels"])
        interaction_stats[key]["category_groups"].extend(m["category_groups"])

    # Table: Interaction statistics per model
    print("\n% LaTeX Table: Interaction Statistics")
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Interaction statistics across models and datasets. "
          "Avg.~Turns is the average number of interaction steps per conversation. "
          "Clar.~\\% is the fraction of conversations where the system asked at least one "
          "clarification question before providing a final answer. "
          "Rel., Tech., and Irrel. indicate the distribution of relevancy labels "
          "assigned to system clarification questions.}")
    print("\\label{tab:interaction_stats}")
    print("\\small")
    print("\\begin{tabular}{@{}llrrrrr@{}}")
    print("\\toprule")
    print("& & & & \\multicolumn{3}{c}{\\textbf{Relevancy (\\%)}} \\\\")
    print("\\cmidrule(lr){5-7}")
    print("\\textbf{Model} & \\textbf{Dataset} & \\textbf{Avg. Turns} & \\textbf{Clar. \\%} & \\textbf{Rel.} & \\textbf{Tech.} & \\textbf{Irrel.} \\\\")
    print("\\midrule")

    for model in TARGET_MODELS:
        display = MODEL_DISPLAY_NAMES[model]
        first = True
        for dataset in ["bird_dev", "spider_test"]:
            key = (model, dataset)
            if key not in interaction_stats:
                continue

            s = interaction_stats[key]
            avg_turns = np.mean(s["n_interactions"]) if s["n_interactions"] else 0
            clar_rate = sum(s["has_clarification"]) / len(s["has_clarification"]) * 100 if s["has_clarification"] else 0

            rel_counts = Counter(s["relevancy_labels"])
            total_rel = sum(rel_counts.values())
            rel_pct = rel_counts.get("Relevant", 0) / total_rel * 100 if total_rel else 0
            tech_pct = rel_counts.get("Technical", 0) / total_rel * 100 if total_rel else 0
            irrel_pct = rel_counts.get("Irrelevant", 0) / total_rel * 100 if total_rel else 0

            prefix = f"\\multirow{{2}}{{*}}{{{display}}}" if first else ""
            first = False
            ds_label = DATASET_NAMES[dataset]

            print(f"{prefix} & {ds_label} & {avg_turns:.2f} & {clar_rate:.1f} & {rel_pct:.1f} & {tech_pct:.1f} & {irrel_pct:.1f} \\\\")

        print("\\midrule" if model != TARGET_MODELS[-1] else "\\bottomrule")

    print("\\end{tabular}")
    print("\\end{table*}")

    # Interaction stats by category group
    print("\n--- Interaction by category group (all models, all datasets, all modes) ---")
    group_interaction = defaultdict(lambda: {"n_interactions": [], "has_clarification": [], "relevancy_labels": []})

    for (model, dataset, cat_use), m in metrics.items():
        for i, group in enumerate(m["category_groups"]):
            group_interaction[group]["n_interactions"].append(m["n_interactions"][i])
            group_interaction[group]["has_clarification"].append(m["has_clarification"][i])

        # Relevancy can't be indexed per-conversation easily, so we track separately
        # We need per-conversation relevancy. Let's re-aggregate from raw data.

    # Re-aggregate per-group relevancy from all_data (we'll do this in the CSV export)
    for group_name in ["Answerable", "Ambiguous", "Unanswerable"]:
        gi = group_interaction[group_name]
        avg_turns = np.mean(gi["n_interactions"]) if gi["n_interactions"] else 0
        clar_rate = sum(gi["has_clarification"]) / len(gi["has_clarification"]) * 100 if gi["has_clarification"] else 0
        n = len(gi["n_interactions"])
        print(f"  {group_name:15s}: n={n:6d}  avg_turns={avg_turns:.2f}  clar_rate={clar_rate:.1f}%")


def export_csv_summaries(metrics, output_dir):
    """Export CSV files for chart generation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV 1: Overall results per model
    with open(output_dir / "overall_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "model_display", "size", "dataset", "recognition", "classification", "ex", "fb", "n"])
        for model in TARGET_MODELS:
            for dataset in ["bird_dev", "spider_test"]:
                key = (model, dataset, "predicted")
                if key not in metrics:
                    continue
                m = metrics[key]
                rec = compute_rate(m["recognition"])
                cls_ = compute_rate(m["classification"])
                ex = compute_rate(m["solved"])
                fb = compute_rate(m["explained"])
                writer.writerow([
                    model, MODEL_DISPLAY_NAMES[model], MODEL_SIZES[model],
                    DATASET_NAMES[dataset],
                    f"{rec*100:.1f}" if rec else "",
                    f"{cls_*100:.1f}" if cls_ else "",
                    f"{ex*100:.1f}" if ex else "",
                    f"{fb*100:.1f}" if fb else "",
                    len(m["recognition"]),
                ])

    # CSV 2: Ablation results
    ablation = defaultdict(lambda: {"solved": [], "explained": []})
    for (model, dataset, cat_use), m in metrics.items():
        for ds in ["bird_dev", "spider_test"]:
            if dataset == ds:
                ablation[(ds, cat_use)]["solved"].extend(m["solved"])
                ablation[(ds, cat_use)]["explained"].extend(m["explained"])

    with open(output_dir / "ablation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "category_mode", "ex", "fb"])
        for dataset in ["bird_dev", "spider_test"]:
            for cat_use in ["ground_truth", "predicted", "no_category"]:
                key = (dataset, cat_use)
                if key not in ablation:
                    continue
                a = ablation[key]
                ex = compute_rate(a["solved"])
                fb = compute_rate(a["explained"])
                writer.writerow([
                    DATASET_NAMES[dataset], cat_use,
                    f"{ex*100:.1f}" if ex else "",
                    f"{fb*100:.1f}" if fb else "",
                ])

    # CSV 3: Per-category results
    cat_metrics = defaultdict(lambda: {"solved": [], "explained": [], "recognition": [], "classification": []})
    for (model, dataset, cat_use), m in metrics.items():
        if cat_use != "predicted":
            continue
        for i, cat_key in enumerate(m["categories"]):
            key = (dataset, cat_key)
            cat_metrics[key]["solved"].append(m["solved"][i])
            cat_metrics[key]["explained"].append(m["explained"][i])
            cat_metrics[key]["recognition"].append(m["recognition"][i])
            cat_metrics[key]["classification"].append(m["classification"][i])

    with open(output_dir / "per_category_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "category", "group", "recognition", "classification", "ex", "fb"])
        for cat_key in CATEGORY_ORDER:
            cat_label = CATEGORY_SHORT_NAMES.get(cat_key, f"{cat_key[0]}")
            cat_answerable = cat_key[0] == "Answerable"
            cat_solvable = cat_key[0] in ["Answerable", "Lexical Vagueness", "Semantic Mapping Ambiguity",
                                           "Structure Ambiguity", "Conflicting Knowledge", "Missing User Knowledge"]
            group = "Answerable" if cat_answerable else ("Ambiguous" if cat_solvable else "Unanswerable")

            for dataset in ["bird_dev", "spider_test"]:
                key = (dataset, cat_key)
                if key not in cat_metrics:
                    continue
                cm = cat_metrics[key]
                rec = compute_rate(cm["recognition"])
                cls_ = compute_rate(cm["classification"])
                ex = compute_rate(cm["solved"])
                fb = compute_rate(cm["explained"])
                writer.writerow([
                    DATASET_NAMES[dataset], cat_label, group,
                    f"{rec*100:.1f}" if rec else "",
                    f"{cls_*100:.1f}" if cls_ else "",
                    f"{ex*100:.1f}" if ex else "",
                    f"{fb*100:.1f}" if fb else "",
                ])

    # CSV 4: Ablation per model (for delta chart)
    with open(output_dir / "ablation_per_model.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "model_display", "dataset", "category_mode", "ex", "fb"])
        for model in TARGET_MODELS:
            for dataset in ["bird_dev", "spider_test"]:
                for cat_use in ["ground_truth", "predicted", "no_category"]:
                    key = (model, dataset, cat_use)
                    if key not in metrics:
                        continue
                    m = metrics[key]
                    ex = compute_rate(m["solved"])
                    fb = compute_rate(m["explained"])
                    writer.writerow([
                        model, MODEL_DISPLAY_NAMES[model],
                        DATASET_NAMES[dataset], cat_use,
                        f"{ex*100:.1f}" if ex else "",
                        f"{fb*100:.1f}" if fb else "",
                    ])

    # CSV 5: Interaction statistics per model
    with open(output_dir / "interaction_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "model_display", "dataset", "avg_turns", "clar_pct", "rel_pct", "tech_pct", "irrel_pct"])
        for model in TARGET_MODELS:
            for dataset in ["bird_dev", "spider_test"]:
                # Aggregate across all category_use modes
                n_interactions = []
                has_clar = []
                rel_labels = []
                for cat_use in ["ground_truth", "predicted", "no_category"]:
                    key = (model, dataset, cat_use)
                    if key not in metrics:
                        continue
                    n_interactions.extend(metrics[key]["n_interactions"])
                    has_clar.extend(metrics[key]["has_clarification"])
                    rel_labels.extend(metrics[key]["relevancy_labels"])

                if not n_interactions:
                    continue

                avg_turns = np.mean(n_interactions)
                clar_rate = sum(has_clar) / len(has_clar) * 100
                rel_counts = Counter(rel_labels)
                total = sum(rel_counts.values())
                rel_pct = rel_counts.get("Relevant", 0) / total * 100 if total else 0
                tech_pct = rel_counts.get("Technical", 0) / total * 100 if total else 0
                irrel_pct = rel_counts.get("Irrelevant", 0) / total * 100 if total else 0

                writer.writerow([
                    model, MODEL_DISPLAY_NAMES[model],
                    DATASET_NAMES[dataset],
                    f"{avg_turns:.2f}", f"{clar_rate:.1f}",
                    f"{rel_pct:.1f}", f"{tech_pct:.1f}", f"{irrel_pct:.1f}",
                ])

    # CSV 6: Interaction by category group
    group_stats = defaultdict(lambda: {"n_interactions": [], "has_clarification": []})
    for (model, dataset, cat_use), m in metrics.items():
        for i, group in enumerate(m["category_groups"]):
            group_stats[(dataset, group)]["n_interactions"].append(m["n_interactions"][i])
            group_stats[(dataset, group)]["has_clarification"].append(m["has_clarification"][i])

    with open(output_dir / "interaction_by_group.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "group", "avg_turns", "clar_pct", "n"])
        for dataset in ["bird_dev", "spider_test"]:
            for group in ["Answerable", "Ambiguous", "Unanswerable"]:
                key = (dataset, group)
                if key not in group_stats:
                    continue
                gs = group_stats[key]
                avg_turns = np.mean(gs["n_interactions"])
                clar_rate = sum(gs["has_clarification"]) / len(gs["has_clarification"]) * 100
                writer.writerow([
                    DATASET_NAMES[dataset], group,
                    f"{avg_turns:.2f}", f"{clar_rate:.1f}",
                    len(gs["n_interactions"]),
                ])

    print(f"\nCSV files saved to {output_dir}/")


def get_balanced_question_counts(dataset, balance_by):
    """Load original questions for a dataset, balance, and return a Counter of selected question keys.

    Uses a Counter instead of a set because the dataset may contain duplicate questions
    (identical text, db_id, and category). The Counter tracks how many of each key
    should be retained so that filtering matches the exact balanced count.
    """
    from utils.balancing import balance_questions, category_key_from_dict, group_key_from_dict

    question_path = DATASET_QUESTION_PATHS.get(dataset)
    db_ids = DATASET_DB_IDS.get(dataset)
    if question_path is None or db_ids is None:
        raise ValueError(f"No question path or db_ids configured for dataset '{dataset}'. "
                         f"Add it to DATASET_QUESTION_PATHS and DATASET_DB_IDS.")

    with open(question_path, "r") as f:
        raw_questions = json.load(f)

    # Filter by db_ids (same logic as do_interaction.py)
    db_ids_set = set(db_ids)
    questions = [q for q in raw_questions if q["db_id"] in db_ids_set]

    key_fn = category_key_from_dict if balance_by == "category" else group_key_from_dict
    balanced = balance_questions(questions, key_fn, seed=42)

    print(f"  {dataset}: {len(questions)} -> {len(balanced)} balanced questions")

    return Counter(
        (q["question"], q["db_id"], q["category"]["name"], q["category"].get("subname"))
        for q in balanced
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze interaction results")
    parser.add_argument("--balanced", action="store_true",
                        help="Filter results to a balanced subset of questions")
    parser.add_argument("--balance_by", type=str, choices=["category", "group"], default="group",
                        help="Balance by 13 categories or 3 groups (default: group)")
    args = parser.parse_args()

    results_dir = "results/interaction"
    output_dir = "charts/results_v16"
    if args.balanced:
        output_dir = f"charts/results_v16_balanced_{args.balance_by}"

    print("Loading interaction results...")
    all_data = load_all_results(results_dir)

    if not all_data:
        print("ERROR: No result files found!")
        sys.exit(1)

    print(f"\nLoaded {len(all_data)} result files")

    # Apply balanced filtering if requested
    if args.balanced:
        print(f"\nApplying balanced filtering ({args.balance_by}-level)...")
        balanced_counts_cache = {}
        # Each (model, dataset, category_use) gets its own Counter so that
        # duplicate questions (same text+db_id+category) are matched the correct
        # number of times independently per category_use mode.
        active_counters = {}

        for i, (model, dataset, conversations) in enumerate(all_data):
            if dataset not in balanced_counts_cache:
                balanced_counts_cache[dataset] = get_balanced_question_counts(dataset, args.balance_by)

            filtered = []
            for c in conversations:
                cat_use = c["category_use"]
                counter_key = (model, dataset, cat_use)
                if counter_key not in active_counters:
                    active_counters[counter_key] = balanced_counts_cache[dataset].copy()

                q_key = (
                    c["question"]["question"],
                    c["question"]["db_id"],
                    c["question"]["category"]["name"],
                    c["question"]["category"].get("subname"),
                )
                if active_counters[counter_key][q_key] > 0:
                    filtered.append(c)
                    active_counters[counter_key][q_key] -= 1

            print(f"  {model} / {DATASET_NAMES.get(dataset, dataset)}: {len(conversations)} -> {len(filtered)} conversations")
            all_data[i] = (model, dataset, filtered)

    # Count per model
    model_counts = Counter(model for model, _, _ in all_data)
    for model in TARGET_MODELS:
        print(f"  {MODEL_DISPLAY_NAMES[model]}: {model_counts.get(model, 0)} files")

    print("\nAggregating metrics...")
    metrics = aggregate_metrics(all_data)

    # Generate all outputs
    print_overall_results_table(metrics)
    print_ablation_table(metrics)
    print_per_category_table(metrics)
    print_per_difficulty_table(metrics)
    print_interaction_analysis(metrics)
    compute_ttr_rtf(all_data)
    export_csv_summaries(metrics, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    if args.balanced:
        print(f"(Balanced mode: {args.balance_by}-level)")
    print("=" * 80)


if __name__ == "__main__":
    main()
