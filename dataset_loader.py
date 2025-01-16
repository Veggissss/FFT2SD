import json
import os
from typing import Literal
from config import SYSTEM_PROMPT, END_OF_PROMPT_MARKER
from datasets import Dataset


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

            # Iterate through template and target JSON entries
            for template_entry, target_entry in zip(template_json, target_json):
                template_entry_str = json.dumps(template_entry)
                target_entry_str = json.dumps(target_entry)

                if model_type == "encoder":
                    template_entry_str = template_entry_str.replace(
                        '"value": null', f'"value": {mask_token}'
                    )

                input_text = SYSTEM_PROMPT.format(
                    input_text=text,
                    template_json=template_entry_str,
                )

                if model_type in ["encoder", "decoder"]:
                    target_text = SYSTEM_PROMPT.format(
                        input_text=text,
                        template_json=target_entry_str,
                    )
                else:
                    target_text = target_entry_str + " " + END_OF_PROMPT_MARKER

                dataset_dict["input"].append(input_text)
                dataset_dict["output"].append(target_text)

    # Convert dict to Hugging Face Dataset.
    dataset = Dataset.from_dict(dataset_dict)

    return dataset


def save_dataset(dataset: Dataset, dataset_path):
    dataset.save_to_disk(dataset_path)
