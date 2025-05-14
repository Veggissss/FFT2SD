import os
import sys
import json

# Add project root directory to sys.path for imports
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(SCRIPT_PATH, "..")))

from pathlib import Path
from utils.file_loader import load_json


def convert_jsons_to_jsonl(input_dir, output_file):
    """
    Convert all JSON files in the input directory to a single JSONL file.
    Each JSON file is expected to contain a list of records.
    For publishing the dataset to Hugging Face.
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)

    json_data_list = []
    for json_file in input_path.glob("*.json"):
        json_data = load_json(json_file)
        json_data_list.append(json_data)
        print(f"Loaded {json_file.name}")

    # Merge all JSON data into a single json list file
    with open(output_path, "w", encoding="utf-8") as f:
        for json_data in json_data_list:
            json.dump(json_data, f, ensure_ascii=False)
            f.write("\n")

    print(
        f"Merged dataset files into {output_path.name} with {len(json_data_list)} entries."
    )


if __name__ == "__main__":
    convert_jsons_to_jsonl("data/corrected/", "data/dataset-eval.jsonl")
    convert_jsons_to_jsonl("data/auto_labeled/", "data/dataset-train.jsonl")
