"""Generate charts from interaction analysis CSV summaries for paper_v3.tex."""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


DATA_DIR = Path("charts/results_v16")
OUTPUT_DIR = Path("charts/results_v16")


def read_csv(filename):
    """Read a CSV file and return list of dicts."""
    with open(DATA_DIR / filename, "r") as f:
        return list(csv.DictReader(f))


def generate_per_category_ex_chart():
    """Generate grouped bar chart of EX and FB by category, split by dataset."""
    rows = read_csv("per_category_results.csv")

    # Separate EX for answerable+ambiguous, FB for unanswerable
    categories_ex = []
    bird_ex = []
    spider_ex = []

    categories_fb = []
    bird_fb = []
    spider_fb = []

    seen_cats_ex = []
    seen_cats_fb = []

    for row in rows:
        cat = row["category"]
        dataset = row["dataset"]
        group = row["group"]

        if group in ("Answerable", "Ambiguous"):
            if cat not in seen_cats_ex:
                seen_cats_ex.append(cat)
                categories_ex.append(cat)
                bird_ex.append(None)
                spider_ex.append(None)
            idx = seen_cats_ex.index(cat)
            val = float(row["ex"]) if row["ex"] else 0
            if dataset == "BIRD":
                bird_ex[idx] = val
            else:
                spider_ex[idx] = val
        else:  # Unanswerable
            if cat not in seen_cats_fb:
                seen_cats_fb.append(cat)
                categories_fb.append(cat)
                bird_fb.append(None)
                spider_fb.append(None)
            idx = seen_cats_fb.index(cat)
            val = float(row["fb"]) if row["fb"] else 0
            if dataset == "BIRD":
                bird_fb[idx] = val
            else:
                spider_fb[idx] = val

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [2.2, 1]})

    # EX chart (answerable + ambiguous)
    x = np.arange(len(categories_ex))
    width = 0.35
    bird_vals = [v if v is not None else 0 for v in bird_ex]
    spider_vals = [v if v is not None else 0 for v in spider_ex]

    ax1.bar(x - width/2, bird_vals, width, label="BIRD", color="#1d3557")
    ax1.bar(x + width/2, spider_vals, width, label="Spider", color="#e63946")
    ax1.set_ylabel("Execution Accuracy (%)", fontsize=14)
    ax1.set_title("Execution Accuracy by Category", fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories_ex, rotation=45, ha='right', fontsize=12)
    ax1.legend(fontsize=16)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(max(bird_vals), max(spider_vals)) * 1.15)

    # FB chart (unanswerable)
    x2 = np.arange(len(categories_fb))
    bird_fb_vals = [v if v is not None else 0 for v in bird_fb]
    spider_fb_vals = [v if v is not None else 0 for v in spider_fb]

    ax2.bar(x2 - width/2, bird_fb_vals, width, label="BIRD", color="#1d3557")
    ax2.bar(x2 + width/2, spider_fb_vals, width, label="Spider", color="#e63946")
    ax2.set_ylabel("Feedback Quality (%)", fontsize=14)
    ax2.set_title("Feedback Quality by Category", fontsize=16, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories_fb, rotation=45, ha='right', fontsize=12)
    ax2.legend(fontsize=16)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_category_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved per_category_performance.png")


def generate_ablation_chart():
    """Generate ablation chart: EX and FB by category mode, per model."""
    rows = read_csv("ablation_per_model.csv")

    models = []
    model_data = defaultdict(lambda: defaultdict(dict))

    for row in rows:
        display = row["model_display"]
        dataset = row["dataset"]
        mode = row["category_mode"]
        ex = float(row["ex"]) if row["ex"] else 0
        fb = float(row["fb"]) if row["fb"] else 0
        if display not in models:
            models.append(display)
        model_data[(display, dataset)][mode] = {"ex": ex, "fb": fb}

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1d3557", "#457b9d", "#a8dadc"]
    mode_labels = {"ground_truth": "Ground Truth", "predicted": "Predicted", "no_category": "No Category"}

    for ax, metric, ylabel, title in [
        (axes[0], "ex", "Execution Accuracy (%)", "EX by Category Mode"),
        (axes[1], "fb", "Feedback Quality (%)", "FB by Category Mode"),
    ]:
        x = np.arange(len(models))
        width = 0.25
        offsets = [-width, 0, width]

        for i, mode in enumerate(["ground_truth", "predicted", "no_category"]):
            # Average across BIRD and Spider
            vals = []
            for m in models:
                bird_val = model_data.get((m, "BIRD"), {}).get(mode, {}).get(metric, 0)
                spider_val = model_data.get((m, "Spider"), {}).get(mode, {}).get(metric, 0)
                vals.append((bird_val + spider_val) / 2)
            ax.bar(x + offsets[i], vals, width, label=mode_labels[mode], color=colors[i])

        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=16)
        ax.legend(fontsize=16)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_by_model.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved ablation_by_model.png")


def generate_interaction_by_group_chart():
    """Generate chart showing interaction patterns by question group."""
    rows = read_csv("interaction_by_group.csv")

    groups = []
    bird_turns = []
    spider_turns = []
    bird_clar = []
    spider_clar = []

    for row in rows:
        group = row["group"]
        if group not in groups:
            groups.append(group)
            bird_turns.append(0)
            spider_turns.append(0)
            bird_clar.append(0)
            spider_clar.append(0)

        idx = groups.index(group)
        if row["dataset"] == "BIRD":
            bird_turns[idx] = float(row["avg_turns"])
            bird_clar[idx] = float(row["clar_pct"])
        else:
            spider_turns[idx] = float(row["avg_turns"])
            spider_clar[idx] = float(row["clar_pct"])

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(groups))
    width = 0.35

    # Avg turns
    ax1.bar(x - width/2, bird_turns, width, label="BIRD", color="#1d3557")
    ax1.bar(x + width/2, spider_turns, width, label="Spider", color="#e63946")
    ax1.set_ylabel("Average Turns", fontsize=14)
    ax1.set_title("Average Conversation Length", fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=16)
    ax1.legend(fontsize=16)
    ax1.grid(axis='y', alpha=0.3)

    # Clarification rate
    ax2.bar(x - width/2, bird_clar, width, label="BIRD", color="#1d3557")
    ax2.bar(x + width/2, spider_clar, width, label="Spider", color="#e63946")
    ax2.set_ylabel("Clarification Rate (%)", fontsize=14)
    ax2.set_title("Clarification Question Rate", fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=16)
    ax2.legend(fontsize=16)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interaction_by_group.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved interaction_by_group.png")


def generate_model_relevancy_chart():
    """Generate chart showing relevancy distribution per model."""
    rows = read_csv("interaction_stats.csv")

    models = []
    rel_data = defaultdict(lambda: {"rel": 0, "tech": 0, "irrel": 0, "count": 0})

    for row in rows:
        display = row["model_display"]
        if display not in models:
            models.append(display)
        # Average across datasets
        rel_data[display]["rel"] += float(row["rel_pct"])
        rel_data[display]["tech"] += float(row["tech_pct"])
        rel_data[display]["irrel"] += float(row["irrel_pct"])
        rel_data[display]["count"] += 1

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.6

    rel_vals = [rel_data[m]["rel"] / rel_data[m]["count"] for m in models]
    tech_vals = [rel_data[m]["tech"] / rel_data[m]["count"] for m in models]
    irrel_vals = [rel_data[m]["irrel"] / rel_data[m]["count"] for m in models]

    bottoms1 = np.array(rel_vals)
    bottoms2 = bottoms1 + np.array(tech_vals)

    ax.bar(x, rel_vals, width, label="Relevant", color="#457b9d")
    ax.bar(x, tech_vals, width, bottom=bottoms1, label="Technical", color="#a8dadc")
    ax.bar(x, irrel_vals, width, bottom=bottoms2, label="Irrelevant", color="#e63946")

    ax.set_ylabel("Distribution (%)", fontsize=14)
    ax.set_title("Relevancy of System Clarification Questions", fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=16)
    ax.legend(fontsize=16)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_relevancy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved model_relevancy.png")


def main():
    global DATA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Generate result charts from CSV summaries")
    parser.add_argument("--balanced", action="store_true",
                        help="Use balanced CSV results")
    parser.add_argument("--balance_by", type=str, choices=["category", "group"], default="group",
                        help="Balance mode (default: group)")
    args = parser.parse_args()

    if args.balanced:
        base = Path(f"charts/results_v16_balanced_{args.balance_by}")
        DATA_DIR = base
        OUTPUT_DIR = base

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating charts from CSV summaries in {DATA_DIR}/...\n")

    generate_per_category_ex_chart()
    generate_ablation_chart()
    generate_interaction_by_group_chart()
    generate_model_relevancy_chart()

    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
