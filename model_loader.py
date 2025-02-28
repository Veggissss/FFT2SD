import json
import torch
from transformers import StoppingCriteriaList

from utils.file_loader import json_to_str
from utils.token_constraints import StopOnToken
from utils.config import SYSTEM_PROMPT, MODELS_DICT
from utils.enums import ModelType
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
        model_type (ModelType): The type of model architecture.
        is_trained (bool): Flag indicating if the model is pre-trained.
        model_name (str): The name of the model.
        device (torch.device): The device to run the model on (CPU or GPU).
        strategy (BaseModelStrategy): The strategy for handling the model based on its type.
        model: The loaded transformer model.
        tokenizer: The tokenizer associated with the model.
        stopping_criteria (StoppingCriteriaList): Criteria to stop the generation process.
    """

    def __init__(
        self,
        model_type: ModelType,
        is_trained: bool,
    ):
        self.model_type = model_type
        self.is_trained = is_trained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use either a trained local model or a Hugging Face model
        if is_trained:
            self.model_name = f"trained/{model_type.value}"
        else:
            self.model_name = MODELS_DICT[model_type.value]

        # Load the model architecture specific handler
        self.strategy: BaseModelStrategy = {
            ModelType.ENCODER_DECODER: EncoderDecoderStrategy,
            ModelType.DECODER: DecoderStrategy,
            ModelType.ENCODER: EncoderStrategy,
        }[model_type]()

        # Load model with corresponding tokenizer
        self.model, self.tokenizer = self.strategy.load(self)

        # Set stopping criteria to json end (Not used for encoder model)
        self.stopping_criteria = StoppingCriteriaList(
            [StopOnToken(self.tokenizer, "}")]
        )

        print(f"Model loaded: {self.model_name}")
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
        self, input_text: str, container_number: str, template_entry: dict
    ) -> dict:
        """
        Fill out the JSON values based on the input text.
        :param model_loader: ModelLoader object with the loaded model and tokenizer.
        :param text: Input text to extract information from.
        :param template_entry: JSON template with value field to be filled.
        :return: JSON with filled values.
        """
        # Convert JSON template to string to be used in generation with enum field
        template_str = json_to_str(template_entry, indent=None)

        # Remove enum field from prompt to save tokens
        if template_entry.get("enum"):
            del template_entry["enum"]

        prompt_template_str = json_to_str(template_entry, indent=None)
        # Convert JSON template to a string to include in the prompt.
        prompt = SYSTEM_PROMPT.format(
            input_text=input_text,
            container_number=container_number,
            # Strip the value field if the model is a decoder speed up generation
            template_json=(
                prompt_template_str.rsplit('"value":', 1)[0] + '"value": '
                if self.model_type == ModelType.DECODER
                else prompt_template_str
            ),
        )
        print(f"Prompt:\n{prompt}\n")

        # Generate the filled JSON based on the prompt
        output_text = self.__generate(prompt, template_str)

        # Convert model output text to JSON
        return self.__output_to_json(output_text)
