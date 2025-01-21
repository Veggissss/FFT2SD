import os
import json
import copy
from typing import Literal
from datasets import Dataset

from config import SYSTEM_PROMPT, END_OF_PROMPT_MARKER


def create_dataset(
    dataset_path: str,
    model_type: Literal["decoder", "encoder", "encoder-decoder"],
    mask_token: str,
) -> Dataset:
    """
    Create a Hugging Face Dataset from the JSON files in the specified directory.
    :param dataset_path: The path to the directory containing the JSON files.
    :param model_type: The type of model architecture - 'encoder', 'encoder-decoder', or 'decoder'.
    :param mask_token: The mask token used for the encoder model.
    :return: Hugging Face Dataset.
    """
    # Dataset dictionary
    dataset_dict = {
        "input": [],
        "output": [],
    }
    for filename in os.listdir(dataset_path):
        if not filename.endswith(".json"):
            print(f"Skipping non-JSON file: {filename}")
            continue

        # Load JSON file and process data
        with open(os.path.join(dataset_path, filename), "r", encoding="utf-8") as f:
            loaded_json_data = json.load(f)

            text = json.dumps(loaded_json_data["input_text"])
            template_json = json.loads(json.dumps(loaded_json_data["template_json"]))
            target_json = json.loads(json.dumps(loaded_json_data["target_json"]))

            container_json = json.loads(json.dumps(loaded_json_data["container_json"]))

            # Add container amount into training data
            target_json.insert(0, copy.deepcopy(container_json[0]))

            # Set "value" fields to null for "antall glass"/container amount into template
            if "value" in container_json[0]:
                container_json[0]["value"] = None
            template_json.insert(0, container_json[0])

            # Iterate through template and target JSON entries
            for template_entry, target_entry in zip(template_json, target_json):
                template_entry_str = json.dumps(template_entry)
                target_entry_str = json.dumps(target_entry)

                # Inject container number into the prompt
                container_number = json.dumps(container_json[1]["value"])

                if model_type == "encoder":
                    template_entry_str = template_entry_str.replace(
                        '"value": null', f'"value": {mask_token}'
                    )

                input_text = SYSTEM_PROMPT.format(
                    input_text=text,
                    container_number=container_number,
                    template_json=template_entry_str,
                )

                if model_type == "decoder":
                    target_text = SYSTEM_PROMPT.format(
                        input_text=text,
                        container_number=container_number,
                        template_json=target_entry_str,
                    )
                elif model_type == "encoder":
                    # Just add the masked value to the target text for the encoder model
                    target_text = json.dumps(target_entry["value"])
                else:
                    target_text = target_entry_str + " " + END_OF_PROMPT_MARKER

                dataset_dict["input"].append(input_text)
                dataset_dict["output"].append(target_text)

    # Convert dict to Hugging Face Dataset.
    return Dataset.from_dict(dataset_dict)


if __name__ == "__main__":
    # Define the path to the dataset directory
    DATASET_PATH = "data/labeled_data/test"
    # Define the model type and mask token
    MODEL_TYPE = "encoder-decoder"

    # Create a Hugging Face Dataset
    dataset = create_dataset(DATASET_PATH, MODEL_TYPE, "[VALUE_MASK]")

    print(dataset)
    print(dataset["input"][0])
    print(dataset["output"][0])

    assert len(dataset["input"]) == len(dataset["output"])
