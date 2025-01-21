import json
import os
import copy
from data_model_enum import get_enum_names

SCRIPT_PATH = os.path.dirname(__file__)


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_json_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(file_path: str, data: dict) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def get_valid_input(input_prompt, item: dict) -> str | float | int | bool:
    try:
        labeled_input = input(input_prompt)

        # If the field is an enum, check if the input is valid
        if "enum" in item["type"]:
            if labeled_input not in item["enum"]:
                # Check if the input is an enum index
                input_index = int(labeled_input)
                if input_index < 0 or input_index >= len(item["enum"]):
                    print(f"Invalid input! Must be one of {item['enum']}")
                    return get_valid_input(input_prompt, item)
                labeled_input = item["enum"][input_index]

        elif "float" in item["type"]:
            labeled_input = float(labeled_input)

        elif "int" in item["type"]:
            labeled_input = int(labeled_input)

        elif "boolean" in item["type"]:
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


def main(
    input_text_name: str,
    input_text_path: str,
    input_json_path: str,
    input_container_path: str,
    output_dir_name: str,
) -> None:
    input_text = read_text_file(input_text_path)
    target_json = read_json_file(input_json_path)
    container_json = read_json_file(input_container_path)

    # Create a non filled out template JSON
    template_json = copy.deepcopy(target_json)
    final_json = {"template_json": template_json}

    print(f"Input text:\n{input_text}\n")

    # Get the number of containers/Beholder-IDs
    while True:
        report_count = get_valid_input(
            "How many containers/Beholder-IDs are there?\n: ", {"type": "int"}
        )
        if report_count < 1:
            print("Invalid input. Please enter a positive integer.")
        else:
            break

    # Update the total amount of containers present in the input text
    container_json[0]["value"] = report_count

    for container_index in range(report_count):
        # Set the container number
        container_json[1]["value"] = container_index + 1

        # For every field in the JSON, fill out the value field
        for item in target_json:
            print(f"Input text:\n{input_text}")
            print(f"Container Number {container_index + 1}:")

            # Update the JSON with the valid labeled input
            input_prompt = f"Enter a value for {item['field']}.\n{item['type']}\n"
            if item.get("enum") is not None:
                for index, enum_name in enumerate(get_enum_names(item["enum"])):
                    if enum_name is not None:
                        input_prompt += (
                            f"{enum_name} [{item['enum'][index]}] (Nummer: {(index)})\n"
                        )
                    else:
                        input_prompt += f"{item['enum'][index]} (Number: {(index)})\n"
            input_prompt += ": "
            item["value"] = get_valid_input(input_prompt, item)

        # Combine input text and target JSON
        final_json["input_text"] = input_text
        final_json["target_json"] = target_json
        final_json["container_json"] = container_json
        print(
            f"JSON Labeled:\n{json.dumps(final_json, indent=4, ensure_ascii=False)}\n"
        )

        # Create output directory if it doesn't exist
        output_dir = os.path.join(SCRIPT_PATH, "labeled_data", output_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save to file
        json_file_name = input_text_name.replace(".txt", ".json")
        output_json_path = f"{output_dir}/container_{container_index}_{json_file_name}"
        write_json_file(output_json_path, final_json)


if __name__ == "__main__":
    # Dir with text files to be labeled
    input_text_dir = os.path.join(SCRIPT_PATH, "example_batch")

    # Generated structured json files from data_model/out/
    input_json_dir = os.path.join(SCRIPT_PATH, "../data_model/out")

    # Output directory name
    output_dir_name = "test"

    # For every text case fill out the JSON values
    for text_file_name in os.listdir(input_text_dir):
        if text_file_name.endswith(".txt"):
            text_path = os.path.join(input_text_dir, text_file_name)

            json_container_path = os.path.join(
                input_json_dir, "generated-beholder.json"
            )

            if "klinisk" in text_file_name or "makro" in text_file_name:
                json_path = os.path.join(input_json_dir, "generated-klinisk.json")

            elif "mikro" in text_file_name:
                json_path = os.path.join(input_json_dir, "generated-makroskopisk.json")

            elif "diagn" in text_file_name:
                json_path = os.path.join(input_json_dir, "generated-mikroskopisk.json")
            else:
                print(f"Could not find a matching JSON file for {text_file_name}.")
                continue

            main(
                text_file_name,
                text_path,
                json_path,
                json_container_path,
                output_dir_name,
            )
