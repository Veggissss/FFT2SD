import os
import json


def load_enum_json(enum_name: str) -> dict:
    """
    Loads all JSON files from the 'enum' directory and returns their contents as a list of dictionaries.

    Returns:
        dict: A list of dictionaries containing the data from each JSON file in the 'enum' directory.
    """
    folder_path = os.path.join(os.path.dirname(__file__), "../data_model/enum")
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            print(f"Skipping non-JSON file: {filename}")
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            for item in data:
                if item.get("value", None) == enum_name:
                    return data

    return {}


def get_enum_fields(enums: list[str], field: str = "name") -> list[str]:
    """
    Retrieves the human readable name of a code enum.
    Args:
        enum (str): The value of the enumeration to look up.
    Returns:
        str: The name of the enumeration if found, otherwise None.
    """
    enum_json = load_enum_json(enums[0])

    result = []
    for entry in enum_json:
        if entry.get(field, None) is None:
            return []
        result.append(entry.get(field))
    return result
