import json
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
    reduce_null_bias,
    add_logits_filter_mask,
    TokenTypeConstraintProcessor,
    StopOnToken,
)
from config import JSON_START_MARKER, HF_TOKEN
from utils.file_loader import str_to_json
from utils.data_classes import TokenOptions

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
            (model_loader.model_settings.peft_model_name or model_loader.model_name),
            trust_remote_code=True,
            token=HF_TOKEN,
        )

    def generate(
        self,
        model_loader: "ModelLoader",
        inputs: dict[str, torch.Tensor],
        amount_new_tokens: int,
        full_template_json: list[dict],
        token_options: TokenOptions,
    ) -> list[str]:
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
        try:
            output_json: list[dict] = str_to_json(
                output_text.split(JSON_START_MARKER)[-1]
            )
        except json.JSONDecodeError:
            print("Failed to parse model output into JSON! Raw output:", output_text)
            output_json = template_entry
            output_json["value"] = None

        # Clean and convert string values to appropriate types
        cleaned_data = self.clean_json(output_json)

        # Add possible enum values back to the output JSON
        if cleaned_data.get("type") == "enum":
            cleaned_data["enum"] = template_entry["enum"]

        return cleaned_data

    def outputs_to_json(
        self,
        output_texts: list[str],
        full_template_json: list[dict],
    ) -> list[dict]:
        """
        Convert the model output text to a JSON object.
        :param outputs: Model output text to convert.
        :param full_template_json: The untokenized full template as a JSON.
        :return: JSON object.
        """
        filled_json_list = []
        for i, output in enumerate(output_texts):
            filled_json_list.append(self.output_to_json(output, full_template_json[i]))
        return filled_json_list

    def get_template_allowed_tokens(
        self,
        full_template_json: list[dict],
        token_options: TokenOptions,
    ) -> list[list[int]]:
        """
        Get the allowed tokens based on the template type.
        :param full_template_json: The untokenized full template as a JSON.
        :param report_type: The report type (KLINISK, MAKROSKOPISK, MIKROSKOPISK) enum or None for caching diabled.
        :return: List of token IDs that are allowed for the given template type.
        """
        # Check for cached allowed tokens
        if (
            token_options.report_type
            and token_options.report_type in self.allowed_tokens_map
        ):
            return self.allowed_tokens_map[token_options.report_type]

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
                    token_options.allow_null,
                )
            )

        # Store the allowed tokens in the map for caching
        if token_options.report_type:
            self.allowed_tokens_map[token_options.report_type] = allowed_token_ids_list

        return allowed_token_ids_list

    def _get_peft_config(
        self, model_loader: "ModelLoader"
    ) -> tuple[str, BitsAndBytesConfig | None]:
        if model_loader.model_settings.use_4bit_quant:
            # Load the base HF model without "_peft" in the name
            model_name = model_loader.model_settings.base_model_name

            # Use 4-bit quantization
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            return model_name, q_config
        if model_loader.model_settings.use_8bit_quant:
            # Load the base HF model without "_peft" in the name
            model_name = model_loader.model_settings.base_model_name

            # Use 8-bit quantization
            q_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            return model_name, q_config

        # If not using quant, return the model name and None
        return model_loader.model_name, None

    def _load_peft_model(
        self, model_loader: "ModelLoader", model: AutoModel
    ) -> AutoModel | PeftModel:
        # Resize to fit the pad. If trained also fit other new tokens from the base model to match the peft model.
        model.resize_token_embeddings(len(self.tokenizer))

        # If not using PEFT, return the model
        if not model_loader.model_settings.use_4bit_quant:
            return model

        if model_loader.is_trained or model_loader.model_settings.peft_model_name:
            # See to use PEFT model from HuggingFace or a locally fine-tuned PEFT model
            if model_loader.model_settings.peft_model_name:
                model_id = model_loader.model_settings.peft_model_name
            else:
                model_id = model_loader.model_name

            # Load the PEFT model
            is_trainable = False
            model = PeftModel.from_pretrained(
                model,
                model_id=model_id,
                # NOTE: Don’t use low_cpu_mem_usage=True when creating a new PEFT adapter for training.
                # https://huggingface.co/docs/peft/v0.15.0/en/package_reference/peft_model#peft.PeftModel.low_cpu_mem_usage
                low_cpu_mem_usage=(not is_trainable),
                is_trainable=is_trainable,
                ephemeral_gpu_offloading=True,
                token=HF_TOKEN,
            )
            # Reduce latency by merging peft model with base model
            if not is_trainable:
                model.eval()
                model.merge_and_unload()
        return model


class EncoderDecoderStrategy(BaseModelStrategy):
    """
    Encoder-Decoder model strategy.
    Sequence-to-sequence model with encoder and decoder.
    """

    def load(self, model_loader: "ModelLoader") -> tuple[AutoModel, AutoTokenizer]:
        # Load tokenizer
        super().load(model_loader)

        # Get possible PEFT config and base model
        model_name, q_config = self._get_peft_config(model_loader)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=q_config,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        model.to(model_loader.device)

        # Load possible PEFT model
        return self._load_peft_model(model_loader, model), self.tokenizer

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        full_template_json,
        token_options: TokenOptions,
    ) -> list[str]:
        logits_processor = LogitsProcessorList(
            [
                TokenTypeConstraintProcessor(
                    self.tokenizer,
                    self.get_template_allowed_tokens(full_template_json, token_options),
                    token_options,
                )
            ]
        )
        stopping_criteria = StoppingCriteriaList([StopOnToken(self.tokenizer, "}")])

        # Add start tokens for the decoder to speed up generation and allow untrained models to work with constrained tokens
        start_tokens = []
        for token in ["{", '"', "value", '"', ":"]:
            token_ids = model_loader.tokenizer(
                token, add_special_tokens=False
            ).input_ids
            start_tokens.extend(token_ids)
        start_tokens = torch.tensor(start_tokens).to(model_loader.device)

        # Make start tokens match the batch size
        start_tokens = start_tokens.expand(inputs["input_ids"].shape[0], -1)

        output_ids = model_loader.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=amount_new_tokens,
            decoder_input_ids=start_tokens,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
        )
        # Delete tensor to free up memory
        del start_tokens

        return model_loader.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def output_to_json(self, output_text: str, template_entry: dict) -> dict:
        # Parse output text to JSON and clean string values
        try:
            output_json = str_to_json(output_text)
        except json.JSONDecodeError:
            print("Failed to parse model output into JSON! Raw output:", output_text)
            output_json = template_entry
            output_json["value"] = None

        cleaned_data = self.clean_json(output_json)

        # Add the output value to the template JSON
        template_entry["value"] = cleaned_data["value"]
        return template_entry


class DecoderStrategy(BaseModelStrategy):
    """
    Decoder model strategy.
    Next token prediction model.
    """

    def load(self, model_loader: "ModelLoader") -> tuple[AutoModel, AutoTokenizer]:
        # Load tokenizer
        super().load(model_loader)
        self.tokenizer.padding_side = "left"

        # Get possible PEFT config and base model
        model_name, q_config = self._get_peft_config(model_loader)

        # Load the base model.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=q_config,
            device_map="auto",
            token=HF_TOKEN,
            # Gemma quant fix: https://huggingface.co/google/gemma-3-4b-it/discussions/41
            torch_dtype=(
                torch.bfloat16 if model_loader.device.type == "cuda" else torch.float16
            ),
            # trust_remote_code=True,
        )

        if (
            self.tokenizer.pad_token is None
            or self.tokenizer.pad_token is self.tokenizer.unk_token
        ):
            # Add pad token if unset
            self.tokenizer.add_special_tokens(
                {
                    "pad_token": "<PAD>",
                }
            )
        else:
            print(f"Using existing pad token: {self.tokenizer.pad_token}")

        # Load possible PEFT model
        return self._load_peft_model(model_loader, model), self.tokenizer

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        full_template_json,
        token_options: TokenOptions,
    ) -> list[str]:
        # Generate decoder output
        inputs.pop("token_type_ids", None)
        logits_processor = LogitsProcessorList(
            [
                TokenTypeConstraintProcessor(
                    self.tokenizer,
                    self.get_template_allowed_tokens(full_template_json, token_options),
                    token_options,
                )
            ]
        )
        # Stopping criteria to stop generation when json is done generating
        stopping_criteria = StoppingCriteriaList([StopOnToken(self.tokenizer, "}")])

        output_ids = model_loader.model.generate(
            **inputs,
            max_new_tokens=amount_new_tokens,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
        )
        return model_loader.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


class EncoderStrategy(BaseModelStrategy):
    """
    Encoder model strategy.
    Random masked token prediction model with constrained unmasking tokens.
    """

    def load(self, model_loader: "ModelLoader") -> tuple[AutoModel, AutoTokenizer]:
        # Load tokenizer
        super().load(model_loader)

        # Get possible PEFT config and base model
        model_name, q_config = self._get_peft_config(model_loader)

        # Encoder model
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            quantization_config=q_config,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        model.to(model_loader.device)

        # Load possible PEFT model
        return self._load_peft_model(model_loader, model), self.tokenizer

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        full_template_json,
        token_options: TokenOptions,
    ) -> list[str]:
        # Forward pass to get logits
        with torch.no_grad():
            outputs = model_loader.model(**inputs)
            # [batch_size, seq_length, vocab_size]
            logits = outputs.logits

            # [batch_size, seq_length]
            input_ids = inputs.input_ids
            batch_size = logits.shape[0]

        # Find masked token positions
        batch_indices, token_indices = (
            input_ids == model_loader.tokenizer.mask_token_id
        ).nonzero(as_tuple=True)
        assert all(batch_indices.bincount() == 1), "Max 1 mask!"

        # Get logits for the masked token [batch_size, vocab_size]
        batch_masked_logits = logits[batch_indices, token_indices]

        # Prepare allowed token IDs for the batch
        allowed_token_ids_batch = self.get_template_allowed_tokens(
            full_template_json, token_options
        )

        # Apply masking and collect predictions
        for i in range(batch_size):
            batch_masked_logits[i] = add_logits_filter_mask(
                batch_masked_logits[i], allowed_token_ids_batch[i]
            )
            batch_masked_logits[i] = reduce_null_bias(
                self.tokenizer, batch_masked_logits[i], token_options.reduce_null_bias
            )
            log_token_probabilities(
                model_loader.tokenizer,
                batch_masked_logits[i],
                allowed_token_ids_batch[i],
            )

        # Replace masks with predicted tokens [batch_size]
        input_ids[batch_indices, token_indices] = torch.argmax(
            batch_masked_logits, dim=1
        )

        # Decode the entire batch
        return model_loader.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
