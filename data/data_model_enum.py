import os
import sys

# Add project root directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from file_loader import load_json


def load_enum_json(enum_name: str) -> dict | None:
    """
    Searches all JSON files from the 'enum' directory and returns the full json containing the given enum.
    :param enum_name: The name of the enum to search for.
    :return: The full JSON containing the enum. Can be an empty dictionary.
    """
    folder_path = os.path.join(os.path.dirname(__file__), "../data_model/enum")
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            print(f"Skipping non-JSON file: {filename}")
            continue

        filepath = os.path.join(folder_path, filename)
        data = load_json(filepath)
        for item in data:
            if item.get("value", None) == enum_name:
                return data
    return None


def get_enum_fields(enum: str, field: str = "name") -> list[str] | None:
    """
    Retrieves the human readable name of a code enum.
    :param enums (str): The enum to look up (example "T65520").
    :param field (str): The field to look up in the enum.
    :return (list[str]): The list of enum fields containing readable terms like "terminale ileum". Can be None.
    """
    enum_json = load_enum_json(enum) or {}

    result = []
    for entry in enum_json:
        if entry.get(field, None) is None:
            return None
        result.append(entry.get(field))
    return result
