import gc
import torch

from utils.file_loader import json_to_str
from utils.enums import ModelType
from utils.data_classes import TemplateGeneration, TokenOptions, ModelSettings
from config import SYSTEM_PROMPT, MODELS_DICT, DEBUG_MODE_ENABLED
from model_strategy import (
    BaseModelStrategy,
    EncoderDecoderStrategy,
    DecoderStrategy,
    EncoderStrategy,
)


class ModelLoader:
    """
    Class to load and generate from a transformer model and its tokenizer with the specified architecture type.
    Params:
        model_type (ModelType): The type of model architecture (encoder, decoder, encoder-decoder).
        model_index (int): The index of the model settings to use from the `config.py` model dictionary. (Generaly increasing model size with index)
        is_trained (bool): Flag indicating if the model is fine-tuned and is saved locally.
        load_model_name (str): The full name of the model to load from Hugging Face or local path with possible suffixes.
    NOTE: Either `model_index` or `load_model_name` should be provided.
    Attributes:
        model_settings: The settings for the model, including its name and other specific configs.
        model_name (str): The name of the model/path to local if is_trained.
        model: The loaded transformer model.
        tokenizer: The tokenizer associated with the model.
        strategy (BaseModelStrategy): The strategy for handling the model based on its type.
        device (torch.device): The device to run the model on (CPU or GPU).
    """

    def __init__(
        self,
        model_type: ModelType = None,
        model_index: None | int = None,
        is_trained: bool = True,
        load_model_name: None | str = None,
    ):
        if load_model_name is not None:
            self.model_type, self.model_index = self.get_model_index(load_model_name)
        elif model_type is not None and model_index is not None:
            self.model_type = model_type
            self.model_index = model_index
        else:
            raise ValueError(
                "Either (model_index and model_type) or (model_name) must be provided."
            )
        self.is_trained = is_trained
        self.model_settings: ModelSettings = MODELS_DICT[self.model_type][
            self.model_index
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use either a trained local model or a Hugging Face model
        if self.is_trained and self.model_settings.is_fine_tuning:
            self.model_name = f"trained/{str(self.model_settings)}"
        else:
            self.model_name = self.model_settings.base_model_name

        # Load the model architecture specific handler
        self.strategy: BaseModelStrategy = {
            ModelType.ENCODER_DECODER: EncoderDecoderStrategy,
            ModelType.DECODER: DecoderStrategy,
            ModelType.ENCODER: EncoderStrategy,
        }[self.model_type]()

        # Load model with corresponding tokenizer
        self.model, self.tokenizer = self.strategy.load(self)

        print(f"Model loaded: {self.model_name}")
        print(f"Device: {self.device}")

    def generate_filled_json(
        self,
        generation: TemplateGeneration,
        token_options: TokenOptions = None,
    ) -> list[dict]:
        """
        Generate filled JSON objects based on the input text and template.

        Args:
            generation (TemplateGeneration): The input text and template JSON to fill.
            token_options (TokenOptions): Optional token options for generation.
        Returns:
            list[dict]: A list of filled JSON objects based on the input text and template.
        """
        if token_options.include_enums:
            # Remove group field from enum (if excists) to save tokens
            for template_entry in generation.template_json:
                if "enum" in template_entry:
                    for enum_entry in template_entry["enum"]:
                        if "group" in enum_entry:
                            del enum_entry["group"]
                            continue
                        # Exit early if group field is not found
                        break
        else:
            # Remove whole enum field from prompt to save tokens
            for template_entry in generation.template_json:
                if "enum" in template_entry:
                    del template_entry["enum"]

        # Convert JSON template to a string to include in the prompt.
        prompts = []
        for template_entry in generation.template_json:
            template_entry_str = json_to_str(template_entry, indent=None)

            prompt = SYSTEM_PROMPT.format(
                input_text=generation.input_text,
                container_id=generation.container_id,
                # Strip the value field if the model is a decoder speed up generation
                template_json=(
                    template_entry_str.rsplit('"value":', 1)[0] + '"value":'
                    if self.model_type == ModelType.DECODER
                    else template_entry_str
                ),
            )
            # print(f"Prompt:\n{prompt}\n")
            prompts.append(prompt)

        # Generate the filled JSON based on the prompt
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        prompt_tokens = inputs["input_ids"].shape[1]
        if DEBUG_MODE_ENABLED:
            print(f"Amount of prompt tokens: {prompt_tokens}")
        # Set max_tokens to the length of the input IDs
        # NOTE: Stopping criteria should stop the generation before hitting the max tokens
        max_tokens = prompt_tokens
        output_texts = self.strategy.generate(
            self,
            inputs,
            max_tokens,
            generation.original_template_json,
            token_options,
        )

        assert (
            len(generation.original_template_json) == inputs["input_ids"].shape[0]
        ), "Batch size mismatch with template length!"

        # Remove the input tensors to free up memory
        del inputs
        torch.cuda.empty_cache()

        # Format every model output text in the batch to JSON
        return self.strategy.outputs_to_json(
            output_texts, generation.original_template_json
        )

    def unload_model(self) -> None:
        """
        Unload the model from memory.
        """
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "strategy"):
            del self.strategy
        if hasattr(self, "model"):
            self.model.to("cpu")
            del self.model

        # Clear CUDA cache
        if torch.cuda.is_available():
            with torch.no_grad():
                torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

        print(f"Model unloaded: {self.model_name}")

    def get_model_index(self, model_name: str) -> tuple[ModelType, int] | None:
        """
        Find the model type and index for the given model_name in the MODELS_DICT.
        Args:
            model_name: The name of the model to find, might include special quant such as "_4bit_quant".
        Returns:
            A tuple containing the model type and index if found, otherwise None.
        """
        for model_type, models in MODELS_DICT.items():
            for i, model_settings in enumerate(models):
                if str(model_settings) == model_name:
                    return model_type, i
        return None
