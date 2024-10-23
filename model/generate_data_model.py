from typing import Union, List, Dict, Any
import json
import os

# Directory where enum JSON files are stored
enum_folder_path = "enums"

enum_replacement_string = "REF_ENUM"
enum_replacement_separator = ";"

def load_json_file(path : str) -> List[Dict[str, Any]]:
    """Load enum values from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def replace_enum_references(data: Union[List[Any], Dict[str, Any]]) -> None:
    """Recursively replace enum references in the data."""
    if isinstance(data, list):
        for item in data:
            replace_enum_references(item)
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and value.startswith(enum_replacement_string):
                # Enum filename without prefix
                enum_name = value.split(enum_replacement_separator)[1]
                enum_file = os.path.join(enum_folder_path, f"{enum_name}.json")
                
                if os.path.exists(enum_file):
                    enum_values = load_json_file(enum_file)

                    # Extract the value field if available, else name field from each enum object
                    data[key] = [enum.get("value", enum.get("name")) for enum in enum_values]
                else:
                    print(f"Warning: Enum file {enum_file} not found. Keeping original value.")
            else:
                replace_enum_references(value)

def main(input_json_file : str, output_json_file : str) -> None:
    """Main function to replace enum values and save the output."""
    # Load the original JSON data
    data = load_json_file(input_json_file)
    
    # Replace enum references
    replace_enum_references(data)

    # Save the modified JSON data
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_json_file = "klinisk.json"      # Change this to your input JSON file
    output_json_file = f"out/generated-{input_json_file}"      # Output file for the modified JSON

    main(input_json_file, output_json_file)
