import os
import copy
from datasets import Dataset
from config import SYSTEM_PROMPT, CONTAINER_NUMBER_MASK
from file_loader import load_json, json_to_str
from enums import ModelType


def create_dataset(dataset_path: str, model_type: ModelType) -> tuple[Dataset, list]:
    """
    Create a Hugging Face Dataset from the JSON files in the specified directory.
    :param dataset_path: The path to the directory containing the JSON files.
    :param model_type: The type of model architecture - 'encoder', 'encoder-decoder', or 'decoder'.
    :return: Hugging Face Dataset.
    """
    # Dataset dictionary
    dataset_dict = {
        "input": [],
        "output": [],
    }
    enums = []
    for filename in os.listdir(dataset_path):
        if not filename.endswith(".json"):
            print(f"Skipping non-JSON file: {filename}")
            continue

        # Load JSON file and process data
        loaded_json_data = load_json(os.path.join(dataset_path, filename))

        # Text to extract information from
        input_text_str = json_to_str(loaded_json_data["input_text"], indent=1)

        target_json = loaded_json_data["target_json"]
        metadata_json = loaded_json_data["metadata_json"]

        # Add total container amount and report type into training data
        target_json.insert(0, copy.deepcopy(metadata_json[0]))
        target_json.insert(1, copy.deepcopy(metadata_json[1]))

        # Create a template JSON with all value fields set to None
        template_json = reset_value_fields(copy.deepcopy(target_json))

        # Iterate through template and target JSON entries
        for template_entry, target_entry in zip(template_json, target_json):
            template_entry_str = json_to_str(template_entry)
            target_entry_str = json_to_str(target_entry)

            # Inject container number into the prompt
            container_number = json_to_str(metadata_json[1]["value"])

            if template_entry.get("field") == "Antall glass":
                # Mask the container number only once
                container_number = CONTAINER_NUMBER_MASK

            if model_type in [ModelType.ENCODER, ModelType.DECODER]:
                # Use target as input
                input_text = SYSTEM_PROMPT.format(
                    input_text=input_text_str,
                    container_number=container_number,
                    template_json=target_entry_str,
                )
                # Not used by the encoder and decoder models
                # As the decoder uses next token prediction
                # And the encoder uses random masked token prediction
                target_text = "[UNUSED]"
            else:
                input_text = SYSTEM_PROMPT.format(
                    input_text=input_text_str,
                    container_number=container_number,
                    template_json=template_entry_str,
                )
                target_text = target_entry_str

            dataset_dict["input"].append(input_text)
            dataset_dict["output"].append(target_text)

            if template_entry.get("type") == "enum":
                for enum in template_entry["enum"]:
                    if not enum:
                        # Replace None with the string "null"
                        enum = "null"
                    if enum not in enums:
                        enums.append(str(enum))

    # Convert dict to Hugging Face Dataset.
    return Dataset.from_dict(dataset_dict), enums


def reset_value_fields(input_json: list[dict], key="value", value=None) -> list[dict]:
    """
    Set all the value fields in the JSON to None.
    """
    for item in input_json:
        item[key] = value
    return input_json
