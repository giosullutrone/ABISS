"""
Semantic distribution visualization: compares generated v16 datasets
against original BIRD/Spider using sentence embeddings + UMAP.

Joint UMAP version: a single UMAP fit on all data so both panels
share the same embedding space and are directly comparable.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer
import umap

# ── Paths ──────────────────────────────────────────────────────────────
BIRD_ORIG = "../datasets/bird_dev/dev.json"
SPIDER_ORIG = [
    "../datasets/spider_test/test.json",
    "../datasets/spider_test/train_spider.json",
]
BIRD_GEN = "example_results/dev_generated_question_v16_merged.json"
SPIDER_GEN = "example_results/spider_test_generated_question_v16_merged.json"
OUT_DIR = "charts/combined_v16"
os.makedirs(OUT_DIR, exist_ok=True)

# ── db_ids used in generation ──────────────────────────────────────────
BIRD_DB_IDS = {
    "california_schools", "card_games", "codebase_community",
    "debit_card_specializing", "european_football_2", "financial",
    "formula_1", "student_club", "superhero", "thrombosis_prediction",
    "toxicology",
}
SPIDER_DB_IDS = {
    "company_1", "company_employee", "company_office", "hr_1",
    "customer_complaints", "customer_deliveries", "customers_and_orders",
    "customers_card_transactions", "department_store", "e_commerce",
    "store_product",
}


def get_question_type(cat: dict) -> str:
    if cat["answerable"]:
        return "Answerable"
    elif cat["solvable"]:
        return "Ambiguous"
    else:
        return "Unanswerable"


def build_text(q: dict, include_evidence: bool) -> str:
    """Concatenate question + evidence for embedding."""
    text = q["question"]
    if include_evidence and q.get("evidence"):
        text += " " + q["evidence"]
    return text


def load_original(paths, db_ids: set, include_evidence: bool):
    if isinstance(paths, str):
        paths = [paths]
    data = []
    for path in paths:
        with open(path) as f:
            data.extend(json.load(f))
    data = [d for d in data if d["db_id"] in db_ids]
    texts = [build_text(d, include_evidence) for d in data]
    return texts


def load_generated(path: str, include_evidence: bool):
    with open(path) as f:
        data = json.load(f)
    texts = [build_text(d, include_evidence) for d in data]
    types = [get_question_type(d["category"]) for d in data]
    return texts, types


def plot_semantic(ax, emb_orig, emb_gen, gen_types, title):
    """Plot UMAP projection on a given axis."""
    type_colors = {
        "Answerable":   "#457b9d",
        "Ambiguous":    "#a8dadc",
        "Unanswerable": "#e63946",
    }

    # Plot generated points first (background), colored by type
    for qtype in ["Unanswerable", "Ambiguous", "Answerable"]:
        mask = np.array([t == qtype for t in gen_types])
        if mask.any():
            ax.scatter(
                emb_gen[mask, 0], emb_gen[mask, 1],
                c=type_colors[qtype], s=6, alpha=0.35,
                label=f"Generated — {qtype}", rasterized=True,
            )

    # Plot original points on top (foreground)
    ax.scatter(
        emb_orig[:, 0], emb_orig[:, 1],
        c="black", s=14, alpha=0.85, marker="x", linewidths=0.9,
        label="Original", rasterized=True,
    )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── BIRD (question + evidence) ─────────────────────────────────────
    print("Loading BIRD data...")
    bird_orig_texts = load_original(BIRD_ORIG, BIRD_DB_IDS, include_evidence=True)
    bird_gen_texts, bird_gen_types = load_generated(BIRD_GEN, include_evidence=True)
    print(f"  Original: {len(bird_orig_texts)}, Generated: {len(bird_gen_texts)}")

    # ── Spider (question only — originals have no evidence) ────────────
    print("Loading Spider data...")
    spider_orig_texts = load_original(SPIDER_ORIG, SPIDER_DB_IDS, include_evidence=False)
    spider_gen_texts, spider_gen_types = load_generated(SPIDER_GEN, include_evidence=False)
    print(f"  Original: {len(spider_orig_texts)}, Generated: {len(spider_gen_texts)}")

    # ── Encode all texts together ──────────────────────────────────────
    all_texts = bird_orig_texts + bird_gen_texts + spider_orig_texts + spider_gen_texts
    print(f"Encoding {len(all_texts)} texts...")
    all_emb = model.encode(all_texts, show_progress_bar=True, batch_size=256)

    # ── Joint UMAP on all embeddings ───────────────────────────────────
    print("Running joint UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
    all_2d = reducer.fit_transform(all_emb)

    # ── Split back into per-benchmark groups ───────────────────────────
    n_bo = len(bird_orig_texts)
    n_bg = len(bird_gen_texts)
    n_so = len(spider_orig_texts)

    bird_orig_2d  = all_2d[:n_bo]
    bird_gen_2d   = all_2d[n_bo : n_bo + n_bg]
    spider_orig_2d = all_2d[n_bo + n_bg : n_bo + n_bg + n_so]
    spider_gen_2d  = all_2d[n_bo + n_bg + n_so :]

    # ── Plot ───────────────────────────────────────────────────────────
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_semantic(ax1, bird_orig_2d, bird_gen_2d, bird_gen_types,
                  f"BIRD (orig={n_bo}, gen={len(bird_gen_texts)})")
    plot_semantic(ax2, spider_orig_2d, spider_gen_2d, spider_gen_types,
                  f"Spider (orig={n_so}, gen={len(spider_gen_texts)})")

    # Shared legend
    handles = [
        mpatches.Patch(color="black", label="Original dataset"),
        mpatches.Patch(color="#457b9d", label="Generated — Answerable"),
        mpatches.Patch(color="#a8dadc", label="Generated — Ambiguous"),
        mpatches.Patch(color="#e63946", label="Generated — Unanswerable"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=16,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(OUT_DIR, "semantic_distribution.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
