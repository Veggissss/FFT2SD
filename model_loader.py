import torch

from utils.file_loader import json_to_str
from utils.enums import ModelType
from utils.data_classes import TemplateGeneration, TokenOptions
from config import SYSTEM_PROMPT, MODELS_DICT, INCLUDE_ENUMS
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
        model_type: ModelType,
        model_index: int = 0,
        is_trained: bool = True,
    ):
        self.model_type = model_type
        self.model_index = model_index
        self.is_trained = is_trained
        self.model_settings = MODELS_DICT[model_type][model_index]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use either a trained local model or a Hugging Face model
        if is_trained:
            self.model_name = f"trained/{self.model_settings.__str__()}"
        else:
            self.model_name = self.model_settings.model_name

        # Load the model architecture specific handler
        self.strategy: BaseModelStrategy = {
            ModelType.ENCODER_DECODER: EncoderDecoderStrategy,
            ModelType.DECODER: DecoderStrategy,
            ModelType.ENCODER: EncoderStrategy,
        }[model_type]()

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
        if not INCLUDE_ENUMS:
            # Remove enum field from prompt to save tokens
            for template_entry in generation.template_json:
                if "enum" in template_entry:
                    template_entry.pop("enum")

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

        # Set max_tokens to the length of the input IDs
        # NOTE: Stopping criteria should setop the generation before hitting the max tokens
        max_tokens = inputs["input_ids"].shape[1]
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

        # Format every model output text in the batch to JSON
        return self.strategy.outputs_to_json(
            output_texts, generation.original_template_json
        )
