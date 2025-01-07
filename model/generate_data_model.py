from typing import Union, List, Dict, Any
import json
import os

# Directory of the script
script_dir = os.path.dirname(__file__)

# Directory where JSON files are stored
struct_dir_path = os.path.join(script_dir, "struct")
enum_dir_path = os.path.join(script_dir, "enum")
out_dir_path = os.path.join(script_dir, "out")

# Enum reference string and separator
enum_replacement_string = "REF_ENUM"
enum_replacement_separator = ";"

# TODO: Change to special LLM token: <TOKEN> etc?
value_placeholder = None


def load_json_file(path: str) -> List[Dict[str, Any]]:
    """Load enum values from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, data: List[Dict[str, Any]]) -> None:
    """Save enum values to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def replace_enum_references(data: Union[List[Any], Dict[str, Any]]) -> None:
    """Recursively replace enum references in the data."""
    if isinstance(data, list):
        for item in data:
            replace_enum_references(item)
    elif isinstance(data, dict):
        data["value"] = value_placeholder
        for key, value in data.items():
            if isinstance(value, str) and value.startswith(enum_replacement_string):
                # Enum filename without prefix
                enum_name = value.split(enum_replacement_separator)[1]
                enum_file = os.path.join(enum_dir_path, f"{enum_name}.json")

                if os.path.exists(enum_file):
                    enum_values = load_json_file(enum_file)

                    # Replace enum reference with enum values
                    # TODO: Optimize enum output while being within LLM token limits?
                    data[key] = [enum for enum in enum_values]
                else:
                    print(f"Warning: Enum file {enum_file} not found!")


def main(input_json_file: str, output_json_file: str) -> None:
    """Main function to replace enum values and save the output."""
    # Load the original JSON data
    data = load_json_file(input_json_file)

    # Replace enum references
    replace_enum_references(data)

    # Save the modified JSON data
    save_json_file(output_json_file, data)


if __name__ == "__main__":
    # Generate data model for all JSON files in struct directory
    for file in os.listdir(struct_dir_path):
        if file.endswith(".json"):
            input_json_file = f"{struct_dir_path}\\{file}"
            output_json_file = f"{out_dir_path}\\generated-{file}"

            # Generate data model
            main(input_json_file, output_json_file)
