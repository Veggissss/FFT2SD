import json


def load_json(filepath: str) -> dict:
    """Load JSON data from a given file."""
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict, filepath: str, indent: int = 4) -> None:
    """Save JSON data to a given file."""
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


def json_to_str(data: dict, indent: None | int = 4) -> str:
    """Convert JSON data to a string."""
    return json.dumps(data, ensure_ascii=False, indent=indent)


def str_to_json(json_str: str) -> dict:
    """Convert a JSON string to a dictionary."""
    return json.loads(json_str)


def load_text(filepath: str) -> str:
    """Load text data as str from a given .txt file."""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()
