import os
import sys
from data_model_enum import get_enum_fields


# Add project root directory to sys.path for imports
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(SCRIPT_PATH, "..")))

from utils.file_loader import load_text_file, load_json, save_json, json_to_str
from utils.enums import ReportType


def get_valid_input(input_prompt, item: dict) -> str | int | bool | None:
    """
    Prompt the user for input and validate it based on the item type.
    """
    try:
        labeled_input = input(input_prompt)

        # Allow for empty input to set it as null
        if labeled_input.lower() in ["", "null", "none"]:
            return None

        # If the field is an enum, check if the input is valid
        match item["type"]:
            case "enum":
                if labeled_input not in item["enum"]:
                    # Check if the input is an enum index
                    input_index = int(labeled_input)
                    if input_index < 0 or input_index >= len(item["enum"]):
                        print(f"Invalid input! Must be one of {item['enum']}")
                        return get_valid_input(input_prompt, item)
                    labeled_input = item["enum"][input_index]["value"]

            case "int":
                labeled_input = int(labeled_input)
                if labeled_input < 0:
                    print("Invalid input! Please enter a positive integer.")
                    return get_valid_input(input_prompt, item)

            case "boolean":
                if labeled_input.lower() not in ["true", "false"]:
                    print("Invalid input! Please enter 'true' or 'false'.")
                    return get_valid_input(input_prompt, item)
                labeled_input = labeled_input.lower() == "true"

    except ValueError:
        # If casts are failed due to invalid input, ask again
        print(f"Invalid input! Please enter an {item['type']}.")
        return get_valid_input(input_prompt, item)

    # Return the valid labeled input
    return labeled_input


def get_labeled_value(item: dict) -> str | int | bool | None:
    """
    Prompts the user to enter a value for a given field based on the item's type
    """
    input_prompt = f"Enter a value for {item['field']}.\n{item['type']}\n"
    if "enum" in item:
        input_prompt += format_enum_options(item["enum"])

    return get_valid_input(input_prompt + ": ", item)


def label_data(
    report_type_name: str,
    input_text_path: str,
    input_json_path: str,
    input_metadata_path: str,
    output_dir_path: str,
) -> None:
    """
    Prompt the user to fill out the JSON values for a given text file.
    The labeled JSON is saved to the output directory.
    """
    input_text = load_text_file(input_text_path)
    target_json = load_json(input_json_path)
    metadata_json = load_json(input_metadata_path)

    print(f"Input text:\n{input_text}\n")

    while True:
        report_count = get_valid_input(
            "How many containers/Beholder-IDs are there?\n: ", {"type": "int"}
        )
        if report_count is None:
            print("Invalid input. Please enter a positive integer.")
            continue
        if report_count > 0:
            break
        print("Invalid input. Please enter a positive integer.")

    metadata_json[0]["value"] = report_type_name
    metadata_json[1]["value"] = report_count

    # For each container, fill out the JSON values
    for container_index in range(1, report_count + 1):
        metadata_json[2]["value"] = container_index
        final_json = {
            "input_text": input_text,
            "target_json": [],
            "metadata_json": metadata_json.copy(),
        }

        for item in target_json:
            print("\n" * 5)
            print(f"Input text:\n{input_text}")
            print(f"Glass nummer {container_index}:")
            item["value"] = get_labeled_value(item)
            final_json["target_json"].append(item)

        save_labeled_json(final_json, input_text_path, output_dir_path, container_index)


def format_enum_options(enum_codes: list[str]) -> str:
    """
    Get a user friendly string representation of the available enum options.
    :param enum_codes: List of codes such as M09401.
    """
    formatted_options = ""
    for index, option in enumerate(enum_codes):
        value = option["value"]  # Value should always be present
        name = option["name"] if "name" in option else ""
        group = option["group"] if "group" in option else None

        if group:
            formatted_options += f"[{group}] "
        if name:
            formatted_options += f"{name} "
        else:
            formatted_options += f"{value} "

        formatted_options += f"(Valg nummer: {index})\n"

    return formatted_options


def save_labeled_json(
    final_json: dict, input_text_path: str, output_dir_path: str, container_index: int
) -> None:
    """
    Save the final labeled JSON to a file in the output directory.
    """
    filename = input_text_path.split("\\")[-1].replace(".txt", "")
    print(f"JSON Labeled:\n{json_to_str(final_json)}\n")
    save_json(final_json, f"{output_dir_path}/{filename}_{container_index}.json")


if __name__ == "__main__":
    # Dir with text files to be labeled
    input_text_dir = os.path.join(SCRIPT_PATH, "example_batch")

    # Generated structured json files from data_model/out/
    input_json_dir = os.path.join(SCRIPT_PATH, "../data_model/out")

    # Output directory
    output_dir = os.path.join(SCRIPT_PATH, "test_data")

    # For every text case fill out the JSON values
    for text_filename in os.listdir(input_text_dir):
        if not text_filename.endswith(".txt"):
            print(f"Skipping non-text file: {text_filename}")
            continue

        json_path = None
        report_name = None
        if "klinisk" in text_filename:
            report_name = ReportType.KLINISK.value
        elif "makro" in text_filename:
            report_name = ReportType.MAKROSKOPISK.value
        elif "diag" in text_filename:
            report_name = ReportType.MIKROSKOPISK.value

        if report_name is None:
            print(f"Could not find a matching report type JSON for {text_filename}.")
            continue

        output_json_path = os.path.join(
            output_dir, text_filename.replace(".txt", "_1.json")
        )
        if os.path.exists(output_json_path):
            print(f"Output file already exists for {text_filename}. Skipping.")
            continue

        label_data(
            report_name,
            os.path.join(input_text_dir, text_filename),
            os.path.join(input_json_dir, f"generated-{report_name}.json"),
            os.path.join(input_json_dir, "generated-metadata.json"),
            output_dir,
        )
