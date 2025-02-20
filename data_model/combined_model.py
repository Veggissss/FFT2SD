import os
import sys

# Directory of the script
SCRIPT_PATH = os.path.dirname(__file__)

# Add project root directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(SCRIPT_PATH, "..")))
from file_loader import load_json, save_json


def load_and_combine_json_files(directory):
    """
    Load and combine JSON files in the specified directory.
    Used for creating a full structure UML diagram/figure.
    param directory: The directory containing generated JSON model files.
    """
    combined_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            data = load_json(filepath)

            # Remove every value field in each JSON
            for entry in data:
                if "id" in entry:
                    del entry["id"]
                if "value" in entry:
                    del entry["value"]

                # Shorten enum list if it's too long to fit in the figure better
                if "enum" in entry:
                    enum_list_as_string = ""
                    enum_list = []
                    for i, enum in enumerate(entry["enum"]):
                        enum_list_as_string += str(enum)

                        if i == len(entry["enum"]) - 1 or len(enum_list_as_string) > 21:
                            enum_list.append(enum_list_as_string)
                            enum_list_as_string = ""
                        elif i < len(entry["enum"]) - 1:
                            enum_list_as_string += ", "

                    entry["enum"] = enum_list

            # Make "generated-name.json" be just "name"
            name = filename.split("-")[1].split(".")[0]
            combined_data[name] = data

    return combined_data


if __name__ == "__main__":
    out_directory = SCRIPT_PATH + "/out"
    output_filepath = SCRIPT_PATH + "/figure/full-structure.json"

    combined_json = load_and_combine_json_files(out_directory)
    save_json(combined_json, output_filepath)
