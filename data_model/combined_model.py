import os
from generate_data_model import save_json_file, load_json_file

# Directory of the script
SCRIPT_PATH = os.path.dirname(__file__)


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
            data = load_json_file(filepath)

            # Remove every value field in each JSON
            for entry in data:
                if "value" in entry:
                    del entry["value"]

            # Make "generated-name.json" be just "name"
            name = filename.split("-")[1].split(".")[0]
            combined_data[name] = data

    return combined_data


if __name__ == "__main__":
    out_directory = SCRIPT_PATH + "/out"
    output_filepath = SCRIPT_PATH + "/figure/full-structure.json"

    combined_json = load_and_combine_json_files(out_directory)
    save_json_file(output_filepath, combined_json)
