import os
import json


def load_all_enums() -> list:
    """
    Loads all JSON files from the 'enum' directory and returns their contents as a list of dictionaries.
    TODO: Fix inefficient way of loading enums. Use field name to load enum file directly. Not critical for now.

    Returns:
        dict: A list of dictionaries containing the data from each JSON file in the 'enum' directory.
    """
    folder_path = os.path.join(os.path.dirname(__file__), "../data_model/enum")
    enum_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                for item in data:
                    enum_data.append(item)
    return enum_data


def get_enum_field(enum: str, field: str) -> str:
    """
    Retrieves the human readable name of a code enum.
    Args:
        enum (str): The value of the enumeration to look up.
    Returns:
        str: The name of the enumeration if found, otherwise None.
    """
    enum_data = load_all_enums()
    for data in enum_data:
        if data.get("value") == enum:
            return data[field]
    return None


def get_enum_fields(enums: list[str], field: str = "name") -> list[str]:
    """
    Retrieves the human readable names of a list of code enums.
    Args:
        enums ([str]): A list of enumeration values to look up.
    Returns:
        [str]: A list of enumeration names if found, otherwise None.
    """
    return [get_enum_field(enum, field) for enum in enums]
