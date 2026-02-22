import argparse
import json
import os
from typing import List


def merge_interaction_files(input_paths: List[str], output_path: str) -> None:
    """
    Merge multiple JSON files containing interaction results into a single file.
    Each file is a Results dict with agent_name, user_name, dataset_name, and conversations.
    The conversations lists are concatenated into a single Results dict.

    Args:
        input_paths: List of file paths to JSON files containing interaction results
        output_path: Path where the merged JSON file will be saved
    """
    all_conversations = []
    agent_name = None
    user_name = None
    dataset_name = None

    for file_path in input_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        try:
            with open(file_path, 'r') as f:
                results = json.load(f)

            if not isinstance(results, dict) or "conversations" not in results:
                print(f"Warning: File {file_path} does not contain a valid Results dict")
                continue

            # Use metadata from the first file, raise if subsequent files differ
            if agent_name is None:
                agent_name = results.get("agent_name")
                user_name = results.get("user_name")
                dataset_name = results.get("dataset_name")
            else:
                if results.get("agent_name") != agent_name:
                    raise ValueError(f"agent_name mismatch in {file_path}: "
                                     f"'{results.get('agent_name')}' vs '{agent_name}'")
                if results.get("user_name") != user_name:
                    raise ValueError(f"user_name mismatch in {file_path}: "
                                     f"'{results.get('user_name')}' vs '{user_name}'")
                if results.get("dataset_name") != dataset_name:
                    raise ValueError(f"dataset_name mismatch in {file_path}: "
                                     f"'{results.get('dataset_name')}' vs '{dataset_name}'")

            conversations = results["conversations"]
            all_conversations.extend(conversations)
            print(f"Loaded {len(conversations)} conversations from {file_path}")

        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from {file_path}: {e}")
            continue
        except Exception as e:
            print(f"Error: Failed to read {file_path}: {e}")
            continue

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write merged results to output file
    merged = {
        "agent_name": agent_name,
        "user_name": user_name,
        "dataset_name": dataset_name,
        "conversations": all_conversations
    }
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=4)

    print(f"\nSuccessfully merged {len(all_conversations)} conversations into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple JSON interaction result files into a single file"
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

    merge_interaction_files(args.input_paths, args.output_path)
