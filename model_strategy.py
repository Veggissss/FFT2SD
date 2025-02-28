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
)
from utils.token_constraints import get_allowed_tokens
from utils.config import JSON_START_MARKER, MODELS_DICT
from utils.file_loader import str_to_json

if TYPE_CHECKING:  # just for type definition
    from model_loader import ModelLoader


class BaseModelStrategy:
    """
    Base class for generation strategies. Every model type (encoder, decoder, encoder-decoder) will implement this.
    """

    def __init__(self):
        # Transfer tokenizer from abstract to implemented load method
        self._tokenizer = None

    @abstractmethod
    def load(self, model_loader: "ModelLoader") -> tuple[AutoModel, AutoTokenizer]:
        """
        Load the model and tokenizer for the specific model type.
        """
        # Load the corresponding model's tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
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

    def output_to_json(self, output_text: str) -> dict:
        """
        Convert generated output text back into just a JSON.
        """
        return str_to_json(output_text.split(JSON_START_MARKER)[-1])


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

        return model, self._tokenizer

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        template_str=None,
    ) -> str:
        output_ids = model_loader.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # eos_token_id=8,
            max_new_tokens=amount_new_tokens,
            stopping_criteria=model_loader.stopping_criteria,
        )
        return model_loader.tokenizer.decode(
            output_ids.squeeze(), skip_special_tokens=True
        )

    def output_to_json(self, output_text: str) -> dict:
        # Add the output value to the template JSON
        return str_to_json(output_text)


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
            bnb_4bit_quant_type="nf4",  # NormalFloat4 (recommended for LLMs)
            bnb_4bit_use_double_quant=True,  # Double quantization
            bnb_4bit_compute_dtype="float16",  # Reduce memory further
        )

        # Load the untrained base model.
        model = AutoModelForCausalLM.from_pretrained(
            MODELS_DICT[model_loader.model_type.value],
            quantization_config=q_config,
            low_cpu_mem_usage=True,
            device_map="auto",  # Use the best device, model.to() not needed
        )

        if model_loader.is_trained:
            # Resize to fit the trained model's token embeddings
            model.resize_token_embeddings(len(self._tokenizer))

            # Load the PEFT model
            model = PeftModel.from_pretrained(
                model,
                model_loader.model_name,
            )

        return model, self._tokenizer

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        template_str=None,
    ) -> str:
        # Generate decoder output
        inputs.pop("token_type_ids", None)

        output_ids = model_loader.model.generate(
            **inputs,
            max_new_tokens=amount_new_tokens,
            stopping_criteria=model_loader.stopping_criteria,
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

        return model, self._tokenizer

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
            logits = outputs.logits

        # Find the masked token position
        masked_index = torch.where(
            inputs.input_ids == model_loader.tokenizer.mask_token_id
        )[1].item()

        # Get logits for the masked token
        masked_token_logits = logits[0, masked_index]

        # Apply softmax to convert logits to probabilities
        probabilities = torch.nn.functional.softmax(masked_token_logits, dim=-1)

        # Filter the allowed tokens' probabilities based on template type
        template_json = str_to_json(template_str)
        template_type = template_json["type"]

        # If the template is an enum, get the allowed tokens
        template_enums = None
        if "enum" in template_json:
            template_enums = [enum["value"] for enum in template_json["enum"]]
        # Get allowed tokens
        allowed_token_ids = get_allowed_tokens(
            model_loader.tokenizer,
            template_type,
            template_enums,
        )

        # Check if allowed tokens are empty, and if so, use the model's output without filtering
        if not allowed_token_ids:
            print("No allowed tokens found! Using the highest probability tokens.")
            allowed_token_ids = probabilities.topk(10).indices.tolist()

        # Get token probabilities for allowed tokens
        allowed_token_probabilities = [
            (token_id, probabilities[token_id].item()) for token_id in allowed_token_ids
        ]

        # Print top-k tokens with their probabilities
        top_k = sorted(allowed_token_probabilities, key=lambda x: x[1], reverse=True)[
            :10
        ]
        print("Top tokens and their probabilities:")
        for token_id, prob in top_k:
            token = model_loader.tokenizer.decode(token_id)
            print(f"Token: {token}, Probability: {prob:.4f}")

        # Select the allowed token with the highest probability
        predicted_token_id = max(allowed_token_probabilities, key=lambda x: x[1])[0]

        # Convert token ID back to token
        predicted_token = model_loader.tokenizer.decode(predicted_token_id)

        # Replace the [MASK] token with the predicted token in the original sequence
        input_ids = inputs.input_ids[0].tolist()
        input_ids[masked_index] = predicted_token_id

        print(f"Predicted token: {predicted_token}")

        # Convert the input_ids with the unmasked id back to text
        return model_loader.tokenizer.decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
