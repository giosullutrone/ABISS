import json
import argparse
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_dataclasses.question import Question, QuestionUnanswerable


def load_questions(file_path: str) -> list[Question]:
    """Load questions from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        # Check if this is an unanswerable question
        if "hidden_knowledge" in item or "is_solvable" in item:
            question = QuestionUnanswerable.from_dict(item)
        else:
            question = Question.from_dict(item)
        questions.append(question)
    
    return questions


def create_pie_chart(data: dict, title: str, output_path: str):
    """Create a pie chart with seaborn styling."""
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by count for consistent ordering
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    sizes = [item[1] for item in sorted_items]
    
    # Calculate percentages to check if any are below 0.5%
    total = sum(sizes)
    percentages = [(size / total) * 100 for size in sizes]
    use_legend = any(p < 1.0 for p in percentages)
    
    # Create pie chart with seaborn colors
    colors = sns.color_palette("husl", len(labels))
    
    if use_legend:
        # Use legend to avoid overlapping text for small slices
        wedges, texts = ax.pie( # type: ignore
            sizes,
            startangle=90,
            colors=colors
        )
        
        # Create labels with percentages for legend
        labels_with_percentages = [f"{label} (n={size}, {pct:.1f}%)" for label, size, pct in zip(labels, sizes, percentages)]
        
        # Add legend
        ax.legend(
            wedges,
            labels_with_percentages,
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=10
        )
    else:
        # Use labels on pie slices (default behavior)
        _, texts, autotexts = ax.pie( # type: ignore
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10}
        )
        
        # Enhance text appearance
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {title} to {output_path}")


def visualize_distributions(questions_file: str, output_dir: str = "."):
    """Generate and save pie charts for style, difficulty, and category distributions."""
    # Load questions
    print(f"Loading questions from {questions_file}...")
    questions = load_questions(questions_file)
    print(f"Loaded {len(questions)} questions")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract distributions
    styles = [q.question_style.value for q in questions]
    difficulties = [q.question_difficulty.value for q in questions]
    categories = []
    for q in questions:
        category_name = q.category.get_name()
        subname = q.category.get_subname()
        if subname:
            categories.append(f"{category_name} - {subname}")
        else:
            categories.append(category_name)
    
    # Count distributions
    style_counts = dict(Counter(styles))
    difficulty_counts = dict(Counter(difficulties))
    category_counts = dict(Counter(categories))
    
    # Print statistics
    print(f"\nStyle distribution: {style_counts}")
    print(f"Difficulty distribution: {difficulty_counts}")
    print(f"Category distribution: {category_counts}")
    
    # Create pie charts
    print("\nGenerating pie charts...")
    create_pie_chart(
        style_counts,
        "Question Style Distribution",
        str(output_path / "style_distribution.png")
    )
    
    create_pie_chart(
        difficulty_counts,
        "Question Difficulty Distribution",
        str(output_path / "difficulty_distribution.png")
    )
    
    create_pie_chart(
        category_counts,
        "Question Category Distribution",
        str(output_path / "category_distribution.png")
    )
    
    print("\nAll visualizations complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pie charts for question style, difficulty, and category distributions"
    )
    parser.add_argument(
        "questions_file",
        type=str,
        help="Path to the JSON file containing questions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the output PNG files (default: current directory)"
    )
    
    args = parser.parse_args()
    
    visualize_distributions(args.questions_file, args.output_dir)


if __name__ == "__main__":
    main()
