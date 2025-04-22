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
    StoppingCriteriaList,
)
from token_constraints import (
    get_allowed_tokens,
    log_token_probabilities,
    add_score_mask,
    TokenTypeConstraintProcessor,
    StopOnToken,
)
from config import JSON_START_MARKER, MODELS_DICT, REDUCE_NULL_BIAS
from utils.file_loader import str_to_json
from utils.enums import ReportType

if TYPE_CHECKING:  # just for type definition
    from model_loader import ModelLoader


class BaseModelStrategy:
    """
    Base class for generation strategies. Every model type (encoder, decoder, encoder-decoder) will implement this.
    """

    def __init__(self):
        # Transfer tokenizer from abstract to implemented load method
        self.tokenizer = None
        self.allowed_tokens_map = {}

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
        full_template_json: list[dict],
        report_type: ReportType,
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

    def output_to_json(self, output_text: str, template_entry: dict) -> dict:
        """
        Convert generated output text back into just a JSON.
        """
        output_json: list[dict] = str_to_json(output_text.split(JSON_START_MARKER)[-1])

        # Clean and convert string values to appropriate types
        cleaned_data = self.clean_json(output_json)

        # Add possible enum values back to the output JSON
        if cleaned_data.get("type") == "enum":
            cleaned_data["enum"] = template_entry["enum"]

        return cleaned_data

    def get_type_allowed_tokens(
        self, full_template_json: list[dict], report_type: ReportType | None
    ) -> list[list[int]]:
        """
        Get the allowed tokens based on the template type.
        :param full_template_json: The untokenized full template as a JSON.
        :param report_type: The report type (KLINISK, MAKROSKOPISK, MIKROSKOPISK) enum or None for caching diabled.
        :return: List of token IDs that are allowed for the given template type.
        """
        # Check for cached allowed tokens
        if report_type and report_type in self.allowed_tokens_map:
            return self.allowed_tokens_map[report_type]

        allowed_token_ids_list = []
        for template_json_entry in full_template_json:
            template_type = template_json_entry["type"]

            # If the template is an enum, get the allowed tokens
            template_enums = None
            if "enum" in template_json_entry:
                template_enums = [enum["value"] for enum in template_json_entry["enum"]]
            # Get allowed tokens
            allowed_token_ids_list.append(
                get_allowed_tokens(
                    self.tokenizer,
                    template_type,
                    template_enums,
                )
            )

        # Store the allowed tokens in the map for caching
        if report_type:
            self.allowed_tokens_map[report_type] = allowed_token_ids_list

        return allowed_token_ids_list


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
        full_template_json,
        report_type,
    ) -> str:
        logits_processor = LogitsProcessorList(
            [
                TokenTypeConstraintProcessor(
                    self.tokenizer,
                    self.get_type_allowed_tokens(full_template_json, report_type),
                )
            ]
        )
        stopping_criteria = StoppingCriteriaList([StopOnToken(self.tokenizer, "}")])

        output_ids = model_loader.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=amount_new_tokens,
            logits_processor=logits_processor,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        return model_loader.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def output_to_json(self, output_text: str, template_entry: dict) -> dict:
        # Parse output text to JSON and clean string values
        cleaned_data = self.clean_json(str_to_json(output_text))

        # Add the output value to the template JSON
        template_entry["value"] = cleaned_data["value"]
        return template_entry


class DecoderStrategy(BaseModelStrategy):
    """
    Decoder model strategy.
    Next token prediction model.
    """

    def load(self, model_loader) -> None:
        # Load tokenizer
        super().load(model_loader)

        if model_loader.model_settings.use_peft:
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,  # Use 4-bit quantization
                bnb_4bit_compute_dtype=torch.float16,
            )

            # Load the untrained base model.
            model = AutoModelForCausalLM.from_pretrained(
                MODELS_DICT[model_loader.model_type][
                    model_loader.model_size
                ].model_name,  # Will not include the '_peft'
                quantization_config=q_config,
                device_map="auto",
                # Don’t use low_cpu_mem_usage=True when creating a new PEFT adapter for training.
                # # https://huggingface.co/docs/peft/v0.15.0/en/package_reference/peft_model#peft.PeftModel.low_cpu_mem_usage
            )

            if model_loader.is_trained:
                # Resize to fit the trained model's token embeddings
                model.resize_token_embeddings(len(self.tokenizer))

                # Load the PEFT model
                model = PeftModel.from_pretrained(
                    model,
                    model_loader.model_name,
                    # ephemeral_gpu_offloading=True,
                    # is_trainable=True,
                )
        else:
            # Load non-quant non-PEFT model
            model = AutoModelForCausalLM.from_pretrained(
                model_loader.model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

        return model, self.tokenizer

    def generate(
        self, model_loader, inputs, amount_new_tokens, full_template_json, report_type
    ) -> str:
        # Generate decoder output
        inputs.pop("token_type_ids", None)
        logits_processor = LogitsProcessorList(
            [
                TokenTypeConstraintProcessor(
                    self.tokenizer,
                    self.get_type_allowed_tokens(full_template_json, report_type),
                )
            ]
        )
        # Set stopping criteria to json end (Not used for encoder model)
        stopping_criteria = StoppingCriteriaList([StopOnToken(self.tokenizer, "}")])

        output_ids = model_loader.model.generate(
            **inputs,
            max_new_tokens=amount_new_tokens,
            logits_processor=logits_processor,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        return model_loader.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


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
        full_template_json,
        report_type,
    ) -> list[str]:
        # Forward pass to get logits
        with torch.no_grad():
            outputs = model_loader.model(**inputs)
            # [batch_size, seq_length, vocab_size]
            logits = outputs.logits

            # [batch_size, seq_length]
            input_ids = inputs.input_ids
            batch_size = logits.shape[0]
        assert len(full_template_json) == batch_size

        # bool mask [batch_size, seq_length]
        is_mask_token = input_ids == model_loader.tokenizer.mask_token_id

        # Find masked indices for the whole batch
        batch_indices, token_indices = is_mask_token.nonzero(as_tuple=True)
        assert all((batch_indices.bincount() == 1)), "Max 1 mask!"

        # Get logits for the masked token [batch_size, vocab_size]
        masked_scores = logits[batch_indices, token_indices]

        # Reduce preference for "null" token
        null_token_id = self.tokenizer.convert_tokens_to_ids("null")
        masked_scores[:, null_token_id] -= REDUCE_NULL_BIAS

        # Prepare allowed token IDs for the batch
        allowed_token_ids_batch = self.get_type_allowed_tokens(
            full_template_json, report_type
        )

        # Apply masking and collect predictions
        masked_scores_masked = []
        for i in range(batch_size):
            allowed_ids = allowed_token_ids_batch[i]
            log_token_probabilities(
                model_loader.tokenizer, masked_scores[i], allowed_ids
            )
            masked_score = add_score_mask(masked_scores[i], allowed_ids)
            masked_scores_masked.append(masked_score)

        # Stack masked scores and get predicted tokens [batch_size, vocab_size]
        masked_scores_stacked = torch.stack(masked_scores_masked, dim=0)

        # Replace masks with predicted tokens [batch_size]
        input_ids[batch_indices, token_indices] = torch.argmax(
            masked_scores_stacked, dim=1
        )

        # Decode the entire batch
        return model_loader.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
