import os
import sys

# Add project root directory to sys.path for imports
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(SCRIPT_PATH, "..")))

from utils.file_loader import load_json, save_json, json_to_str
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
    if "unit" in item:
        input_prompt += f"Enhet: {item['unit']}\n"

    return get_valid_input(input_prompt + ": ", item)


def label_data(
    report_type_name: str,
    input_text: str,
    case_id: str,
    input_json_path: str,
    input_metadata_path: str,
    output_dir_path: str,
) -> None:
    """
    Prompt the user to fill out the JSON values for a given text file.
    The labeled JSON is saved to the output directory.
    """
    target_json = load_json(input_json_path)
    metadata_json = load_json(input_metadata_path)

    input_text = input_text.replace("\r", "\n").strip()

    print(f"Input text:\n{input_text}\n")

    while True:
        report_count = get_valid_input(
            "How many containers/samples are there?\n: ", {"type": "int"}
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
            print(f"\n|| Glass nummer {container_index} ||\n")
            item["value"] = get_labeled_value(item)
            final_json["target_json"].append(item)

        save_labeled_json(
            final_json, report_type_name, case_id, container_index, output_dir_path
        )


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
    final_json: dict,
    report_type_name: str,
    case_id: str,
    container_index: int,
    output_dir_path: str,
) -> None:
    """
    Save the final labeled JSON to a file in the output directory.
    """
    print(f"JSON Labeled:\n{json_to_str(final_json)}\n")
    save_json(
        final_json,
        f"{output_dir_path}\\{report_type_name}_{case_id}_{container_index}.json",
    )


if __name__ == "__main__":
    # Generated structured json files from data_model/out/
    input_json_dir = os.path.join(SCRIPT_PATH, "../data_model/out")

    # Output directory
    output_dir = os.path.join(SCRIPT_PATH, "labeled_data")

    # Load the large batch JSON data
    dataset_json: list[dict] = load_json(
        os.path.join(SCRIPT_PATH, "large_batch/export_2025-03-17.json")
    )

    # Calculate 10% of the dataset ~40 samples
    sample_size = int(len(dataset_json) * 0.1)
    dataset_json = dataset_json[:sample_size]
    print(f"Sampling {sample_size} cases from the dataset.")

    # Store the labeled ids to avoid re-labeling the same cases
    ids_json_path = os.path.join(SCRIPT_PATH, "large_batch/labeled_ids.json")
    labeled_ids_json: dict = load_json(ids_json_path)

    for dataset_case in dataset_json:
        if dataset_case["id"] in labeled_ids_json:
            continue

        label_data(
            ReportType.KLINISK.value,
            dataset_case["kliniske_opplysninger"],
            dataset_case["id"],
            os.path.join(input_json_dir, f"generated-{ReportType.KLINISK.value}.json"),
            os.path.join(input_json_dir, "generated-metadata.json"),
            output_dir,
        )
        label_data(
            ReportType.MAKROSKOPISK.value,
            dataset_case["makrobeskrivelse"],
            dataset_case["id"],
            os.path.join(
                input_json_dir, f"generated-{ReportType.MAKROSKOPISK.value}.json"
            ),
            os.path.join(input_json_dir, "generated-metadata.json"),
            output_dir,
        )

        # Combine mikroskopisk and diagnose text
        micro_text = dataset_case["mikrobeskrivelse"]
        if dataset_case["diagnose"] is not None:
            micro_text += "\n" + dataset_case["diagnose"]

        label_data(
            ReportType.MIKROSKOPISK.value,
            micro_text,
            dataset_case["id"],
            os.path.join(
                input_json_dir, f"generated-{ReportType.MIKROSKOPISK.value}.json"
            ),
            os.path.join(input_json_dir, "generated-metadata.json"),
            output_dir,
        )

        # Finished labeling case
        print(f"Finished labeling case {dataset_case['id']}")
        labeled_ids_json[dataset_case["id"]] = {
            "klinisk": True,
            "makroskopisk": True,
            "mikroskopisk": True,
        }
        save_json(labeled_ids_json, ids_json_path)
