"""Comprehensive dataset analysis for paper_v3.tex."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dataset_dataclasses.question import Question, QuestionUnanswerable


def load_questions(file_path: str) -> list:
    with open(file_path, 'r') as f:
        data = json.load(f)
    questions = []
    for item in data:
        if "hidden_knowledge" in item or "is_solvable" in item:
            questions.append(QuestionUnanswerable.from_dict(item))
        else:
            questions.append(Question.from_dict(item))
    return questions


def get_category_label(q):
    name = q.category.get_name()
    subname = q.category.get_subname()
    return f"{name} - {subname}" if subname else name


def get_group_label(q):
    if q.category.is_answerable():
        return "Answerable"
    elif q.category.is_solvable():
        return "Ambiguous"
    else:
        return "Unanswerable"


def analyze_dataset(questions, name):
    print(f"\n{'='*70}")
    print(f"  DATASET: {name}")
    print(f"{'='*70}")
    print(f"Total questions: {len(questions)}")

    # Category group distribution
    groups = Counter(get_group_label(q) for q in questions)
    print(f"\nQuestion type distribution:")
    for g in ["Answerable", "Ambiguous", "Unanswerable"]:
        c = groups.get(g, 0)
        print(f"  {g}: {c} ({c/len(questions)*100:.1f}%)")

    # Category distribution
    cats = Counter(get_category_label(q) for q in questions)
    print(f"\nCategory distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(questions)*100:.1f}%)")

    # Style distribution
    styles = Counter(q.question_style.value for q in questions)
    print(f"\nStyle distribution:")
    for s, c in sorted(styles.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c} ({c/len(questions)*100:.1f}%)")

    # Difficulty distribution
    diffs = Counter(q.question_difficulty.value for q in questions)
    print(f"\nDifficulty distribution:")
    for d, c in sorted(diffs.items(), key=lambda x: -x[1]):
        print(f"  {d}: {c} ({c/len(questions)*100:.1f}%)")

    # Per-database distribution
    dbs = Counter(q.db_id for q in questions)
    print(f"\nPer-database distribution ({len(dbs)} databases):")
    for db, c in sorted(dbs.items(), key=lambda x: -x[1]):
        print(f"  {db}: {c} ({c/len(questions)*100:.1f}%)")

    # Question length analysis
    lengths = [len(q.question.split()) for q in questions]
    print(f"\nQuestion length (words):")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
    print(f"  Std: {np.std(lengths):.1f}")

    # Per-category question length
    cat_lengths = defaultdict(list)
    for q in questions:
        cat_lengths[get_group_label(q)].append(len(q.question.split()))
    print(f"\nQuestion length by type:")
    for g in ["Answerable", "Ambiguous", "Unanswerable"]:
        lens = cat_lengths.get(g, [])
        if lens:
            print(f"  {g}: mean={np.mean(lens):.1f}, median={np.median(lens):.1f}")

    return {
        'total': len(questions),
        'groups': groups,
        'categories': cats,
        'styles': styles,
        'difficulties': diffs,
        'databases': dbs,
        'lengths': lengths,
    }


def generate_combined_bar_chart(bird_stats, spider_stats, output_dir):
    """Generate combined category distribution bar chart."""
    sns.set_style("whitegrid")

    # Get all categories from both datasets
    all_cats = sorted(set(list(bird_stats['categories'].keys()) + list(spider_stats['categories'].keys())))

    # Shorten category names
    short_names = {
        'Answerable - With Evidence': 'Ans. w/ Ev.',
        'Answerable - Without Evidence': 'Ans. w/o Ev.',
        'Conflicting Knowledge': 'Confl. Know.',
        'Improper Question': 'Improper Q.',
        'Lexical Vagueness': 'Lex. Vague.',
        'Missing External Knowledge': 'Miss. Ext. K.',
        'Missing Schema Elements - Missing Entities or Attributes': 'Miss. Sch. Ent.',
        'Missing Schema Elements - Missing Relationships': 'Miss. Sch. Rel.',
        'Missing User Knowledge': 'Miss. User K.',
        'Semantic Mapping Ambiguity - Entity Ambiguity': 'Sem. Entity',
        'Semantic Mapping Ambiguity - Lexical Overlap': 'Sem. Lexical',
        'Structure Ambiguity - Attachment Ambiguity': 'Struct. Attach.',
        'Structure Ambiguity - Scope Ambiguity': 'Struct. Scope',
    }

    labels = [short_names.get(c, c) for c in all_cats]
    bird_vals = [bird_stats['categories'].get(c, 0) for c in all_cats]
    spider_vals = [spider_stats['categories'].get(c, 0) for c in all_cats]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, bird_vals, width, label=f"BIRD ({bird_stats['total']})", color="#1d3557")
    bars2 = ax.bar(x + width/2, spider_vals, width, label=f"Spider ({spider_stats['total']})", color="#e63946")

    ax.set_ylabel('Number of Questions', fontsize=15)
    ax.set_title('Category Distribution: BIRD vs Spider', fontsize=17, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=17)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved category comparison chart to {output_dir}/category_comparison.png")


def generate_group_pie_charts(bird_stats, spider_stats, output_dir):
    """Generate side-by-side pie charts for question type groups."""
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'Answerable': '#457b9d',
              'Ambiguous': '#a8dadc',
              'Unanswerable': '#e63946'}
    order = ['Answerable', 'Ambiguous', 'Unanswerable']

    for ax, stats, title in [(ax1, bird_stats, f"BIRD (n={bird_stats['total']})"),
                              (ax2, spider_stats, f"Spider (n={spider_stats['total']})")]:

        sizes = [stats['groups'].get(g, 0) for g in order]
        cols = [colors[g] for g in order]
        total = sum(sizes)
        labels_pct = [f"{g}\n({s}, {s/total*100:.1f}%)" for g, s in zip(order, sizes)]
        ax.pie(sizes, labels=labels_pct, colors=cols, startangle=90, textprops={'fontsize': 14})
        ax.set_title(title, fontsize=17, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/question_type_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved question type distribution to {output_dir}/question_type_distribution.png")


def generate_difficulty_style_heatmap(questions, output_path, title):
    """Generate a heatmap of difficulty vs style."""
    sns.set_style("whitegrid")
    style_order = ['formal', 'colloquial', 'imperative', 'interrogative', 'descriptive', 'concise']
    diff_order = ['simple', 'moderate', 'complex', 'highly_complex']

    matrix = np.zeros((len(diff_order), len(style_order)), dtype=int)
    for q in questions:
        si = style_order.index(q.question_style.value)
        di = diff_order.index(q.question_difficulty.value)
        matrix[di][si] += 1

    from matplotlib.colors import LinearSegmentedColormap
    palette_cmap = LinearSegmentedColormap.from_list("palette", ["#f1faee", "#a8dadc", "#457b9d", "#1d3557"])

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap=palette_cmap,
                annot_kws={'fontsize': 14},
                xticklabels=[s.capitalize() for s in style_order],
                yticklabels=[d.replace('_', ' ').capitalize() for d in diff_order],
                ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Question Style', fontsize=17)
    ax.set_ylabel('SQL Difficulty', fontsize=17)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_path}")


def generate_per_db_category_chart(questions, output_path, title):
    """Generate stacked bar chart of categories per database."""
    sns.set_style("whitegrid")

    db_cat = defaultdict(lambda: Counter())
    for q in questions:
        db_cat[q.db_id][get_group_label(q)] += 1

    dbs = sorted(db_cat.keys(), key=lambda d: sum(db_cat[d].values()), reverse=True)
    order = ['Answerable', 'Ambiguous', 'Unanswerable']
    colors_list = ['#457b9d', '#a8dadc', '#e63946']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(dbs))
    bottoms = np.zeros(len(dbs))

    for g, col in zip(order, colors_list):
        vals = [db_cat[db].get(g, 0) for db in dbs]
        ax.bar(x, vals, bottom=bottoms, label=g, color=col, width=0.6)
        bottoms += np.array(vals)

    ax.set_ylabel('Number of Questions', fontsize=17)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    # Shorten db names
    short_dbs = [d.replace('_', ' ').title()[:18] for d in dbs]
    ax.set_xticklabels(short_dbs, rotation=45, ha='right', fontsize=15)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=16)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-db chart to {output_path}")


def print_latex_table(bird_stats, spider_stats):
    """Print LaTeX table for paper."""
    all_cats = sorted(set(list(bird_stats['categories'].keys()) + list(spider_stats['categories'].keys())))

    short = {
        'Answerable - With Evidence': 'Answerable w/ Evidence',
        'Answerable - Without Evidence': 'Answerable w/o Evidence',
        'Conflicting Knowledge': 'Conflicting Knowledge',
        'Improper Question': 'Improper Question',
        'Lexical Vagueness': 'Lexical Vagueness',
        'Missing External Knowledge': 'Missing External Knowledge',
        'Missing Schema Elements - Missing Entities or Attributes': 'Missing Schema Entities',
        'Missing Schema Elements - Missing Relationships': 'Missing Schema Relationships',
        'Missing User Knowledge': 'Missing User Knowledge',
        'Semantic Mapping Ambiguity - Entity Ambiguity': 'Semantic Mapping (Entity)',
        'Semantic Mapping Ambiguity - Lexical Overlap': 'Semantic Mapping (Lexical)',
        'Structure Ambiguity - Attachment Ambiguity': 'Structural (Attachment)',
        'Structure Ambiguity - Scope Ambiguity': 'Structural (Scope)',
    }

    print("\n\n% LaTeX TABLE FOR PAPER")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Dataset statistics for the generated BIRD and Spider benchmarks.}")
    print("\\label{tab:dataset_stats}")
    print("\\small")
    print("\\begin{tabular}{@{}lrrrr@{}}")
    print("\\toprule")
    print("\\textbf{Category} & \\multicolumn{2}{c}{\\textbf{BIRD}} & \\multicolumn{2}{c}{\\textbf{Spider}} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    print(" & \\textit{n} & \\textit{\\%} & \\textit{n} & \\textit{\\%} \\\\")
    print("\\midrule")

    for cat in all_cats:
        bc = bird_stats['categories'].get(cat, 0)
        sc = spider_stats['categories'].get(cat, 0)
        bp = bc / bird_stats['total'] * 100
        sp = sc / spider_stats['total'] * 100
        label = short.get(cat, cat)
        print(f"{label} & {bc} & {bp:.1f} & {sc} & {sp:.1f} \\\\")

    print("\\midrule")
    print(f"\\textbf{{Total}} & \\textbf{{{bird_stats['total']}}} & \\textbf{{100.0}} & \\textbf{{{spider_stats['total']}}} & \\textbf{{100.0}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Also print style and difficulty tables
    print("\n\n% STYLE TABLE")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Distribution of question styles across datasets.}")
    print("\\label{tab:style_stats}")
    print("\\small")
    print("\\begin{tabular}{@{}lrrrr@{}}")
    print("\\toprule")
    print("\\textbf{Style} & \\multicolumn{2}{c}{\\textbf{BIRD}} & \\multicolumn{2}{c}{\\textbf{Spider}} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    print(" & \\textit{n} & \\textit{\\%} & \\textit{n} & \\textit{\\%} \\\\")
    print("\\midrule")
    for s in ['formal', 'colloquial', 'imperative', 'interrogative', 'descriptive', 'concise']:
        bc = bird_stats['styles'].get(s, 0)
        sc = spider_stats['styles'].get(s, 0)
        bp = bc / bird_stats['total'] * 100
        sp = sc / spider_stats['total'] * 100
        print(f"{s.capitalize()} & {bc} & {bp:.1f} & {sc} & {sp:.1f} \\\\")
    print("\\midrule")
    print(f"\\textbf{{Total}} & \\textbf{{{bird_stats['total']}}} & \\textbf{{100.0}} & \\textbf{{{spider_stats['total']}}} & \\textbf{{100.0}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Difficulty table
    print("\n\n% DIFFICULTY TABLE")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Distribution of SQL difficulty levels across datasets.}")
    print("\\label{tab:difficulty_stats}")
    print("\\small")
    print("\\begin{tabular}{@{}lrrrr@{}}")
    print("\\toprule")
    print("\\textbf{Difficulty} & \\multicolumn{2}{c}{\\textbf{BIRD}} & \\multicolumn{2}{c}{\\textbf{Spider}} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    print(" & \\textit{n} & \\textit{\\%} & \\textit{n} & \\textit{\\%} \\\\")
    print("\\midrule")
    for d in ['simple', 'moderate', 'complex', 'highly_complex']:
        bc = bird_stats['difficulties'].get(d, 0)
        sc = spider_stats['difficulties'].get(d, 0)
        bp = bc / bird_stats['total'] * 100
        sp = sc / spider_stats['total'] * 100
        label = d.replace('_', ' ').capitalize()
        print(f"{label} & {bc} & {bp:.1f} & {sc} & {sp:.1f} \\\\")
    print("\\midrule")
    print(f"\\textbf{{Total}} & \\textbf{{{bird_stats['total']}}} & \\textbf{{100.0}} & \\textbf{{{spider_stats['total']}}} & \\textbf{{100.0}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    bird_path = "example_results/dev_generated_question_v16_merged.json"
    spider_path = "example_results/spider_test_generated_question_v16_merged.json"
    output_dir = "charts/combined_v16"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bird_qs = load_questions(bird_path)
    spider_qs = load_questions(spider_path)

    bird_stats = analyze_dataset(bird_qs, "BIRD (v16)")
    spider_stats = analyze_dataset(spider_qs, "Spider (v16)")

    # Combined stats
    print(f"\n{'='*70}")
    print(f"  COMBINED")
    print(f"{'='*70}")
    print(f"Total questions: {len(bird_qs) + len(spider_qs)}")

    # Generate charts
    generate_combined_bar_chart(bird_stats, spider_stats, output_dir)
    generate_group_pie_charts(bird_stats, spider_stats, output_dir)
    generate_difficulty_style_heatmap(bird_qs, f"{output_dir}/bird_diff_style_heatmap.png", "BIRD: Difficulty vs Style Distribution")
    generate_difficulty_style_heatmap(spider_qs, f"{output_dir}/spider_diff_style_heatmap.png", "Spider: Difficulty vs Style Distribution")
    generate_per_db_category_chart(bird_qs, f"{output_dir}/bird_per_db.png", "BIRD: Questions per Database")
    generate_per_db_category_chart(spider_qs, f"{output_dir}/spider_per_db.png", "Spider: Questions per Database")

    # Print LaTeX tables
    print_latex_table(bird_stats, spider_stats)
