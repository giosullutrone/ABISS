import argparse
import json
import os
from typing import List


def merge_question_files(input_paths: List[str], output_path: str) -> None:
    """
    Merge multiple JSON files containing questions into a single file.
    
    Args:
        input_paths: List of file paths to JSON files containing questions
        output_path: Path where the merged JSON file will be saved
    """
    all_questions = []
    
    for file_path in input_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r') as f:
                questions = json.load(f)
                
            if not isinstance(questions, list):
                print(f"Warning: File {file_path} does not contain a list of questions")
                continue
                
            all_questions.extend(questions)
            print(f"Loaded {len(questions)} questions from {file_path}")
            
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from {file_path}: {e}")
            continue
        except Exception as e:
            print(f"Error: Failed to read {file_path}: {e}")
            continue
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write merged questions to output file
    with open(output_path, 'w') as f:
        json.dump(all_questions, f, indent=4)
    
    print(f"\nSuccessfully merged {len(all_questions)} questions into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple JSON question files into a single file"
    )
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs='+',
        required=True,
        help="List of input JSON file paths to merge"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged JSON file"
    )
    
    args = parser.parse_args()
    
    merge_question_files(args.input_paths, args.output_path)
