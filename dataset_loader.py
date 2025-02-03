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
) -> tuple[Dataset, list]:
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
    enums = []
    for filename in os.listdir(dataset_path):
        if not filename.endswith(".json"):
            print(f"Skipping non-JSON file: {filename}")
            continue

        # Load JSON file and process data
        with open(os.path.join(dataset_path, filename), "r", encoding="utf-8") as f:
            loaded_json_data = json.load(f)

            # Text to extract information from
            input_text_str = json.dumps(
                loaded_json_data["input_text"], ensure_ascii=False
            )

            template_json = loaded_json_data["template_json"]
            target_json = loaded_json_data["target_json"]
            container_json = loaded_json_data["container_json"]

            # Add container amount into training data
            target_json.insert(0, copy.deepcopy(container_json[0]))
            template_json.insert(0, copy.deepcopy(container_json[0]))

            # Iterate through template and target JSON entries
            for template_entry, target_entry in zip(template_json, target_json):
                # Mask out the correct value in the template JSON
                template_entry["value"] = mask_token

                template_entry_str = json.dumps(template_entry, ensure_ascii=False)
                target_entry_str = json.dumps(target_entry, ensure_ascii=False)

                # Inject container number into the prompt
                container_number = json.dumps(
                    container_json[1]["value"], ensure_ascii=False
                )

                input_text = SYSTEM_PROMPT.format(
                    input_text=input_text_str,
                    container_number=container_number,
                    template_json=template_entry_str,
                    decoder_start="",
                )

                if model_type in ["decoder", "encoder"]:
                    input_text = SYSTEM_PROMPT.format(
                        input_text=input_text_str,
                        container_number=container_number,
                        template_json=target_entry_str,
                        decoder_start=("{" if model_type == "decoder" else ""),
                    )
                    # Not used by the encoder and decoder models
                    # As the decoder uses next token prediction
                    # And the encoder uses random masked token prediction
                    target_text = "[UNUSED]"
                else:
                    target_text = target_entry_str + " " + END_OF_PROMPT_MARKER

                dataset_dict["input"].append(input_text)
                dataset_dict["output"].append(target_text)

                if template_entry.get("type") == "enum":
                    for enum in template_entry["enum"]:
                        if str(enum) not in enums:
                            enums.append(str(enum))

    # Convert dict to Hugging Face Dataset.
    return Dataset.from_dict(dataset_dict), enums
