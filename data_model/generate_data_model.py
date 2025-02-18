from typing import Union, List, Dict, Any
import json
import os


# Enum reference string and separator
ENUM_IDENTIFIER = "REF_ENUM"
ENUM_SEPARATOR = ";"

# TODO: Change to special LLM token: <TOKEN> etc?
VALUE_PLACEHOLDER = None

# Directory of the script
SCRIPT_PATH = os.path.dirname(__file__)

# Directory where JSON files are stored
STRUCT_DIR_PATH = os.path.join(SCRIPT_PATH, "struct")
ENUM_DIR_PATH = os.path.join(SCRIPT_PATH, "enum")
OUT_DIR_PATH = os.path.join(SCRIPT_PATH, "out")


def load_json(filepath: str) -> List[Dict[str, Any]]:
    """Load enum values from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Dict[str, Any]], filepath: str, indent: int = 4) -> None:
    """Save enum values to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def replace_enum_references(data: Union[List[Any], Dict[str, Any]]) -> None:
    """Recursively replace enum references in the data."""
    if isinstance(data, list):
        for item in data:
            replace_enum_references(item)
    elif isinstance(data, dict):
        data["value"] = VALUE_PLACEHOLDER
        for key, value in data.items():
            if isinstance(value, str) and value.startswith(ENUM_IDENTIFIER):
                # Enum filename without prefix
                enum_name = value.split(ENUM_SEPARATOR)[1]
                enum_file = os.path.join(ENUM_DIR_PATH, f"{enum_name}.json")

                if os.path.exists(enum_file):
                    enum_values = load_json(enum_file)

                    # Replace enum reference with enum values
                    # TODO: Optimize enum output while being within LLM token limits?
                    data[key] = [
                        enum.get("value", enum.get("name")) for enum in enum_values
                    ]
                else:
                    print(f"Warning: Enum file {enum_file} not found!")


def generate_data_model(input_json_file: str, output_json_file: str) -> None:
    """Main function to replace enum values and save the output."""
    # Load the original JSON data
    data = load_json(input_json_file)

    # Replace enum references
    replace_enum_references(data)

    # Save the modified JSON data
    save_json(data, output_json_file)


if __name__ == "__main__":
    # Generate data model for all JSON files in struct directory
    for file in os.listdir(STRUCT_DIR_PATH):
        if file.endswith(".json"):
            # Generate data model
            generate_data_model(
                f"{STRUCT_DIR_PATH}\\{file}", f"{OUT_DIR_PATH}\\generated-{file}"
            )
