from abc import abstractmethod
from typing import TYPE_CHECKING
import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BitsAndBytesConfig,
    LogitsProcessorList,
)
from token_constraints import (
    get_allowed_tokens,
    log_token_probabilities,
    add_score_mask,
    TokenTypeConstraintProcessor,
)
from utils.config import JSON_START_MARKER, MODELS_DICT, REDUCE_NULL_BIAS
from utils.file_loader import str_to_json

if TYPE_CHECKING:  # just for type definition
    from model_loader import ModelLoader


class BaseModelStrategy:
    """
    Base class for generation strategies. Every model type (encoder, decoder, encoder-decoder) will implement this.
    """

    def __init__(self):
        # Transfer tokenizer from abstract to implemented load method
        self.tokenizer = None

    @abstractmethod
    def load(self, model_loader: "ModelLoader") -> tuple[AutoModel, AutoTokenizer]:
        """
        Load the model and tokenizer for the specific model type.
        """
        # Load the corresponding model's tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_loader.model_name, trust_remote_code=True
        )

    def generate(
        self,
        model_loader: "ModelLoader",
        inputs: dict[str, torch.Tensor],
        amount_new_tokens: int,
        template_str: str = None,
    ) -> str:
        """
        Generate model output text based on the input prompt.
        :param model_loader: The model loader object.
        :param inputs: Tokenized input prompt.
        :param amount_new_tokens: The number of tokens to generate for autoregressive models.
        :param template_str: The untokenized template as a JSON string. Used for loading datatype and enum constraints for tokens (encoder model).
        :return: Generated output text.
        """
        raise NotImplementedError

    def clean_json(self, output_json: dict) -> dict:
        """
        Clean the JSON output by removing leading/trailing spaces and converting string values to appropriate types.
        """
        cleaned_data = {}
        for key, value in output_json.items():
            if isinstance(key, str):
                key = key.strip()
            if isinstance(value, str):
                value = value.strip()
                # Convert null strings back to None
                type_mapping = {"null": None, "true": True, "false": False}
                if value in type_mapping:
                    value = type_mapping[value]
                elif value.isdigit():
                    value = int(value)
            cleaned_data[key] = value
        return cleaned_data

    def output_to_json(self, output_text: str, template_str: str) -> dict:
        """
        Convert generated output text back into just a JSON.
        """
        output_json = str_to_json(output_text.split(JSON_START_MARKER)[-1])

        # Clean and convert string values to appropriate types
        cleaned_data = self.clean_json(output_json)

        # Add possible enum values back to the output JSON
        if cleaned_data.get("type") == "enum":
            cleaned_data["enum"] = str_to_json(template_str)["enum"]

        return cleaned_data

    def get_type_allowed_tokens(self, template_str: str) -> list[int]:
        """
        Get the allowed tokens based on the template type.
        :param template_str: The untokenized template as a JSON string.
        :return: List of token IDs that are allowed for the given template type.
        """
        template_json = str_to_json(template_str)
        template_type = template_json["type"]

        # If the template is an enum, get the allowed tokens
        template_enums = None
        if "enum" in template_json:
            template_enums = [enum["value"] for enum in template_json["enum"]]
        # Get allowed tokens
        return get_allowed_tokens(
            self.tokenizer,
            template_type,
            template_enums,
        )


class EncoderDecoderStrategy(BaseModelStrategy):
    """
    Encoder-Decoder model strategy.
    Sequence-to-sequence model with encoder and decoder.
    """

    def load(self, model_loader: "ModelLoader") -> None:
        # Load tokenizer
        super().load(model_loader)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_loader.model_name, trust_remote_code=True
        )
        model.to(model_loader.device)

        return model, self.tokenizer

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        template_str=None,
    ) -> str:
        logits_processor = LogitsProcessorList(
            [
                TokenTypeConstraintProcessor(
                    self.tokenizer, self.get_type_allowed_tokens(template_str)
                )
            ]
        )
        output_ids = model_loader.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=amount_new_tokens,
            logits_processor=logits_processor,
            stopping_criteria=model_loader.stopping_criteria,
        )
        return model_loader.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def output_to_json(self, output_text: str, template_str: str) -> dict:
        # Parse output text to JSON and clean string values
        cleaned_data = self.clean_json(str_to_json(output_text))

        # Add the output value to the template JSON
        template_json = str_to_json(template_str)
        template_json["value"] = cleaned_data["value"]
        return template_json


class DecoderStrategy(BaseModelStrategy):
    """
    Decoder model strategy.
    Next token prediction model.
    """

    def load(self, model_loader) -> None:
        # Load tokenizer
        super().load(model_loader)

        q_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 4-bit quantization
        )

        # Load the untrained base model.
        model = AutoModelForCausalLM.from_pretrained(
            MODELS_DICT[model_loader.model_type.value],
            quantization_config=q_config,
            device_map="auto",  # Use the best device, model.to() not needed
        )

        if model_loader.is_trained:
            # Resize to fit the trained model's token embeddings
            model.resize_token_embeddings(len(self.tokenizer))

            # Load the PEFT model
            model = PeftModel.from_pretrained(
                model,
                model_loader.model_name,
            )

        return model, self.tokenizer

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        template_str=None,
    ) -> str:
        # Generate decoder output
        inputs.pop("token_type_ids", None)
        logits_processor = LogitsProcessorList(
            [
                TokenTypeConstraintProcessor(
                    self.tokenizer, self.get_type_allowed_tokens(template_str)
                )
            ]
        )
        output_ids = model_loader.model.generate(
            **inputs,
            max_new_tokens=amount_new_tokens,
            stopping_criteria=model_loader.stopping_criteria,
            logits_processor=logits_processor,
        )
        return model_loader.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=False,
        )


class EncoderStrategy(BaseModelStrategy):
    """
    Encoder model strategy.
    Random masked token prediction model with constrained unmasking tokens.
    """

    def load(self, model_loader):
        # Load tokenizer
        super().load(model_loader)

        # Encoder model
        model = AutoModelForMaskedLM.from_pretrained(
            model_loader.model_name, trust_remote_code=True
        )
        model.to(model_loader.device)

        return model, self.tokenizer

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        template_str=None,
    ) -> str:
        # Forward pass to get logits
        with torch.no_grad():
            outputs = model_loader.model(**inputs)
            # [batch_size, seq_length, vocab_size]
            logits = outputs.logits

        # Find the masked token position
        masked_index = torch.where(
            inputs.input_ids == model_loader.tokenizer.mask_token_id
        )[1].item()

        # Get logits for the masked token [batch_size, vocab_size]
        masked_scores = logits[:, masked_index]

        # Decrease preference for null token
        null_token_id = self.tokenizer.convert_tokens_to_ids("null")
        masked_scores[:, null_token_id] -= REDUCE_NULL_BIAS

        # Get allowed tokens based on the template
        allowed_token_ids = self.get_type_allowed_tokens(template_str)

        # Log allowed token probabilities
        log_token_probabilities(
            model_loader.tokenizer, masked_scores, allowed_token_ids
        )

        # Add a mask to the scores based on the allowed token IDs
        masked_scores = add_score_mask(masked_scores, allowed_token_ids)

        # Inject the predicted token back into the input
        # inputs.input_ids: [batch_size, seq_length]
        input_ids = inputs.input_ids[0]
        input_ids[masked_index] = torch.argmax(masked_scores).item()

        return model_loader.tokenizer.decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
