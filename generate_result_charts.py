"""Generate the per-category performance chart from interaction analysis CSV summaries."""
import argparse
import csv
from pathlib import Path

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
    """Generate grouped bar chart of EX and FB by category, with GT vs Predicted paired bars."""
    rows = read_csv("per_category_results.csv")

    # Collect data keyed by (category, dataset, mode)
    data_ex = {}
    data_fb = {}
    categories_ex = []
    categories_fb = []
    seen_ex = []
    seen_fb = []

    for row in rows:
        cat = row["category"]
        dataset = row["dataset"]
        group = row["group"]
        mode = row["mode"]

        if group in ("Answerable", "Ambiguous"):
            if cat not in seen_ex:
                seen_ex.append(cat)
                categories_ex.append(cat)
            val = float(row["ex"]) if row["ex"] else 0
            data_ex[(cat, dataset, mode)] = val
        else:  # Unanswerable
            if cat not in seen_fb:
                seen_fb.append(cat)
                categories_fb.append(cat)
            val = float(row["fb"]) if row["fb"] else 0
            data_fb[(cat, dataset, mode)] = val

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={'width_ratios': [2.2, 1]})

    # Bar layout: 4 bars per category, grouped as [BIRD-GT, BIRD-Pred | Spider-GT, Spider-Pred]
    bar_w = 0.18
    offsets = [-0.30, -0.10, 0.10, 0.30]
    colors = ["#1d3557", "#6b9ac4", "#c1121f", "#e8787f"]
    labels = ["BIRD (GT)", "BIRD (Pred.)", "Spider (GT)", "Spider (Pred.)"]
    bar_configs = [
        ("BIRD", "ground_truth"),
        ("BIRD", "predicted"),
        ("Spider", "ground_truth"),
        ("Spider", "predicted"),
    ]

    def _plot_grouped_bars(ax, categories, data, ylabel, title):
        x = np.arange(len(categories))
        for i, (ds, mode) in enumerate(bar_configs):
            vals = [data.get((cat, ds, mode), 0) for cat in categories]
            ax.bar(x + offsets[i], vals, bar_w, label=labels[i], color=colors[i])
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    _plot_grouped_bars(ax1, categories_ex, data_ex,
                       "Execution Accuracy (%)", "Execution Accuracy by Category")
    all_ex = [data_ex.get((c, ds, m), 0) for c in categories_ex for ds, m in bar_configs]
    ax1.set_ylim(0, max(all_ex) * 1.15 if all_ex else 100)

    _plot_grouped_bars(ax2, categories_fb, data_fb,
                       "Feedback Quality (%)", "Feedback Quality by Category")
    ax2.set_ylim(0, 105)

    # Single shared legend
    handles, leg_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='upper center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUTPUT_DIR / "per_category_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved per_category_performance.png")


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

    print(f"\nChart saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
