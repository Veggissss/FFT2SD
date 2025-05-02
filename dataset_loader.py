import os
import copy
from datasets import Dataset
from utils.file_loader import load_json, json_to_str
from utils.enums import ModelType
from config import (
    SYSTEM_PROMPT,
    CONTAINER_ID_MASK,
    DATA_MODEL_OUTPUT_FOLDER,
)


class DatasetLoader:
    """
    Class for loading and processing datasets for training and inference.
    It handles the creation of input-output pairs based on the model type and the provided dataset path.
    """

    def __init__(self, model_type: ModelType, mask: str | None = None):
        """
        Initialize the DatasetLoader with an optional tokenizer.
        :param model_type: Model type to create dataset for.
        :param tokenizer: The tokenizer to use for dataset processing
        """
        self.mask = mask
        self.model_type = model_type
        self.dataset = None
        self.enums = None

    def create_dataset(
        self, dataset_path: str, include_enums: bool = False
    ) -> tuple[Dataset, list]:
        """
        Create a Hugging Face Dataset from the labeled JSON files in the dataset path.
        :param dataset_path: Path to the dataset files.
        :param include_enums: Whether to include the possible enum values inside the prompt.
        :return: Tuple of the dataset with "input" and "output" columns and the enum values.
        """
        dataset_dict = {"input": [], "output": []}
        enums = []
        for filename in filter(lambda f: f.endswith(".json"), os.listdir(dataset_path)):
            file_path = os.path.join(dataset_path, filename)
            processed_data = self._process_json_file(file_path, enums, include_enums)
            dataset_dict["input"].extend(processed_data["input"])
            dataset_dict["output"].extend(processed_data["output"])

        if not include_enums:
            # Use enum definitions to create data pairs
            for filename in filter(
                lambda f: f.endswith(".json"), os.listdir(DATA_MODEL_OUTPUT_FOLDER)
            ):
                file_path = os.path.join(DATA_MODEL_OUTPUT_FOLDER, filename)
                processed_data = self._process_enum_file(file_path)
                dataset_dict["input"].extend(processed_data["input"])
                dataset_dict["output"].extend(processed_data["output"])

        self.dataset = Dataset.from_dict(dataset_dict)
        self.enums = enums
        return self.dataset, self.enums

    def _process_json_file(
        self, file_path: str, enums: list, include_enums: bool
    ) -> dict:
        """
        Process a single JSON file and extract input-output pairs for the dataset.
        """
        dataset_entries = {"input": [], "output": []}

        data = load_json(file_path)
        input_text = json_to_str(data["input_text"], indent=1)
        metadata_json = data["metadata_json"]
        target_json = [copy.deepcopy(item) for item in metadata_json[:2]] + data[
            "target_json"
        ]

        # Create a copy with values reset to [MASK] or None
        template_json = reset_value_fields(copy.deepcopy(target_json), value=self.mask)

        for template_entry, target_entry in zip(template_json, target_json):
            if template_entry.get("type") == "enum":
                # Save all unique enum values
                for enum in template_entry["enum"]:
                    enum_value = str(enum.get("value"))
                    # Replace python None with JSON null value
                    if enum_value == "None":
                        enum_value = "null"
                    if enum_value not in enums:
                        enums.append(enum_value)
                # Remove the enum form input and output to save a lot of tokens/max_length
                if not include_enums:
                    if "enum" in template_entry:
                        del template_entry["enum"]
                    if "enum" in target_entry:
                        del target_entry["enum"]

            # Remove the value field from antall glass as it should be not given in the input
            if template_entry.get("field") == "Antall glass":
                container_id = CONTAINER_ID_MASK
            else:
                container_id = json_to_str(metadata_json[1]["value"])

            self._add_prompt_entry(
                dataset_entries,
                input_text,
                container_id,
                target_entry["value"],
                target_entry,
                template_entry,
            )

        return dataset_entries

    def _add_prompt_entry(
        self,
        dataset_dict: dict,
        input_text: str,
        container_id: str,
        correct_value: str,
        target_entry: dict,
        template_entry: dict,
    ) -> None:
        """
        Format and add the input-output pair to the dataset based on the model type.
        Encoder-Decoder: input has the prompt + json with null value and the output target text is { "value": "correct_value" }.
        Decoder and Encoder: input has the prompt + json with correct value, output target text is not used.
        """
        input_prompt = SYSTEM_PROMPT.format(
            input_text=input_text,
            container_id=container_id,
            template_json=json_to_str(
                (
                    target_entry
                    if self.model_type in [ModelType.DECODER, ModelType.ENCODER]
                    else template_entry
                ),
                indent=None,
            ),
        )

        if self.model_type == ModelType.ENCODER_DECODER:
            target_text = json_to_str({"value": correct_value}, indent=None)
        elif self.model_type == ModelType.ENCODER:
            target_text = SYSTEM_PROMPT.format(
                input_text=input_text,
                container_id=container_id,
                template_json=json_to_str(template_entry, indent=None),
            )
        else:
            target_text = f"[UNUSED BY THE {self.model_type.value.upper()} TYPE]"

        dataset_dict["input"].append(input_prompt)
        dataset_dict["output"].append(target_text)

    def _process_enum_file(self, file_path: str) -> dict:
        """
        Process a single JSON file to extract enum-based dataset entries.
        """
        dataset_entries = {"input": [], "output": []}

        target_json = load_json(file_path)
        template_json = reset_value_fields(copy.deepcopy(target_json))

        for template_entry, target_entry in zip(template_json, target_json):
            if template_entry.get("type") != "enum":
                continue

            for enum_dict in template_entry["enum"]:
                prompt = ""
                if enum_dict.get("group"):
                    prompt += f"[{enum_dict['group']}] "
                if enum_dict.get("name"):
                    prompt += enum_dict["name"]
                # If enum has no group or name, use the value as prompt
                if prompt == "":
                    prompt = enum_dict["value"]

                correct_enum = enum_dict["value"]

                if "enum" in template_entry:
                    del template_entry["enum"]
                if "enum" in target_entry:
                    del target_entry["enum"]

                self._add_prompt_entry(
                    dataset_entries,
                    prompt,
                    "1",
                    correct_enum,
                    target_entry,
                    template_entry,
                )

        return dataset_entries


def reset_value_fields(input_json: list[dict], key="value", value=None) -> list[dict]:
    """
    Set all value fields in the JSON to None.
    """
    for item in input_json:
        item[key] = value
    return input_json
