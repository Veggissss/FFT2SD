import json
from typing import Literal
from transformers import StoppingCriteriaList
import torch

from token_constraints import StopOnToken
from config import SYSTEM_PROMPT
from model_strategy import (
    BaseModelStrategy,
    EncoderDecoderStrategy,
    DecoderStrategy,
    EncoderStrategy,
)


class ModelLoader:
    """
    Class to load and generate from a transformer model and its tokenizer with the specified architecture type.

    Attributes:
        model_name (str): The name of the model to load.
        model_type (Literal["decoder", "encoder", "encoder-decoder"]): The type of the model architecture.
        device (torch.device): The device to run the model on (CPU or GPU).
        model (AutoModel): The loaded transformer model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
    """

    def __init__(
        self,
        model_name: str,
        model_type: Literal["decoder", "encoder", "encoder-decoder"],
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model architecture specific handler
        self.strategy: BaseModelStrategy = {
            "encoder-decoder": EncoderDecoderStrategy,
            "decoder": DecoderStrategy,
            "encoder": EncoderStrategy,
        }[model_type]()

        # Load model with corresponding tokenizer
        self.model, self.tokenizer = self.strategy.load(self)

        # Set stopping criteria to json end [Not used for encoder model]
        self.stopping_criteria = StoppingCriteriaList(
            [StopOnToken(self.tokenizer, "}")]
        )

        print(f"Model loaded: {model_name}")
        print(f"Device: {self.device}")

    def __generate(self, prompt: str, template_str: str) -> str:
        """Generate model output based on the input prompt.
        :param prompt: Text input prompt for the model.
        :return: Generated output text.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        tokenized_template = self.tokenizer(
            template_str,
            return_tensors="pt",
        )

        # Check prompt size
        amount_prompt_tokens = inputs["input_ids"].shape[1]
        print(f"Prompt tokens: {amount_prompt_tokens}")

        # Get the amount of new tokens to generate based on the json template
        # Might not be needed as the stopping criteria can be used to stop generation early
        amount_new_tokens = tokenized_template["input_ids"].shape[1]
        print(f"Max new decoder tokens: {amount_new_tokens}")

        # Stop when the closing curly brace is generated

        # Generate output based on the strategy
        return self.strategy.generate(self, inputs, amount_new_tokens, template_str)

    def __output_to_json(self, output_text: str) -> dict:
        """
        Convert the model output text to a JSON object.
        :param output_text: Model output text to convert.
        :return: JSON object.
        """
        filled_json = {}
        try:
            filled_json = self.strategy.output_to_json(output_text)
        except json.JSONDecodeError:
            print("Failed to parse model output into JSON. Raw output:", output_text)

        return filled_json

    def generate_filled_json(
        self, input_text: str, container_number: str, template_str: str
    ) -> dict:
        """
        Fill out the JSON values based on the input text.
        :param model_loader: ModelLoader object with the loaded model and tokenizer.
        :param text: Input text to extract information from.
        :param json_template: JSON template with fields to be filled.
        :return: JSON with filled values.
        """
        # Convert JSON template to a string to include in the prompt.
        prompt = SYSTEM_PROMPT.format(
            input_text=input_text,
            container_number=container_number,
            # Strip the value field if the model is a decoder speed up generation
            template_json=(
                template_str.split('"value":')[0] + '"value": '
                if self.model_type == "decoder"
                else template_str
            ),
        )

        # Generate the filled JSON based on the prompt
        output_text = self.__generate(prompt, template_str)

        # Convert model output text to JSON
        return self.__output_to_json(output_text)
