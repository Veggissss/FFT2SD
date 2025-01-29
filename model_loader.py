from typing import Literal
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForMaskedLM,
    BitsAndBytesConfig,
)
import torch
import json
from allowed_tokens import get_allowed_tokens
from config import SYSTEM_PROMPT, END_OF_PROMPT_MARKER


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
        self.model, self.tokenizer = self.__load_model(model_name, model_type)
        print(f"Model loaded: {model_name}")
        print(f"Device: {self.device}")

    def __load_model(
        self, model_name: str, model_type: str
    ) -> tuple[AutoModel, AutoTokenizer]:
        """
        Load a model based on the specified type.
        :param model_name: The Hugging Face model name.
        :param model_type: Type of model - 'encoder', 'encoder-decoder', or 'decoder'.
        :return: model and tokenizer.
        """
        # Load the corresponding model's tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if model_type == "encoder-decoder":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=True
            )
            model.to(self.device)

        elif model_type == "decoder":
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,  # Use 4-bit quantization
                bnb_4bit_quant_type="nf4",  # NormalFloat4 (recommended for LLMs)
                bnb_4bit_use_double_quant=True,  # Double quantization
                bnb_4bit_compute_dtype="float16",  # Reduce memory further
            )

            device_map = {
                "model.layers": 1,
                "model.embed_tokens": 1,
                "model.norm": 0,
                "lm_head": 1,
            }

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=q_config,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )  # .to(self.device)

            print(model.hf_device_map)
        else:
            model = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True
            )
            model.to(self.device)

        # Set special tokens for the tokenizer
        # tokenizer.unk_token = "<unk>"
        tokenizer.pad_token = "<pad>"
        # tokenizer.eos_token = "</s>"
        # tokenizer.mask_token = "[MASK]"

        # tokenizer.add_special_tokens(
        #    {
        #        "unk_token": tokenizer.unk_token,
        #        "pad_token": tokenizer.pad_token,
        #        "mask_token": tokenizer.mask_token,
        #        "eos_token": tokenizer.eos_token,
        #    }
        # )

        # Resize the token embeddings to match the tokenizer
        # model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate model output based on the input prompt.
        :param prompt: Text input prompt for the model.
        :return: Generated output text.
        """
        inputs = self.tokenizer(
            prompt,
            # padding=True,
            return_tensors="pt",
        ).to(self.device)

        # The prompt is always longer than the output
        amount_new_tokens = min(inputs["input_ids"].shape[1], max_length)
        print(f"Prompt tokenized length: {amount_new_tokens}")

        if self.model_type == "encoder":
            # Generate encoder output
            output = self.model(**inputs)

            # Get the logits from the model output
            logits = output.logits

            # Find the position of the masked token
            masked_position = (
                inputs.input_ids == self.tokenizer.mask_token_id
            ).nonzero(as_tuple=True)[0]

            if masked_position.numel() == 0:
                print("No masked token found in the input.")
                return

            # There is only one masked position
            pos = masked_position[0].item()

            # Get the logits for the masked position and batch size is 1
            position_logits = logits[0, pos]

            allowed_token_ids = get_allowed_tokens(self.tokenizer, "int")
            if len(allowed_token_ids) == 0:
                print("No allowed tokens found!")
                return

            # Create a mask for allowed tokens
            allowed_mask = torch.zeros_like(position_logits, dtype=torch.bool)
            for token_id in allowed_token_ids:
                allowed_mask[token_id] = True

            # Mask the logits of disallowed tokens
            masked_logits = torch.where(
                allowed_mask,
                position_logits,
                torch.tensor(float("-inf")).to(self.device),
            )

            # Get the top 5 logits and their corresponding token IDs from the allowed tokens
            top_k_logits, top_k_ids = torch.topk(masked_logits, 5, dim=-1)

            # Convert logits to probabilities using softmax
            top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)

            print(f"DEBUG Masked position {pos}:")
            for j in range(5):
                token_id = top_k_ids[j].item()
                token_prob = top_k_probs[j].item()
                token_str = self.tokenizer.decode([token_id])
                print(
                    f"  Token ID: {token_id}, Token: '{token_str}', Probability: {token_prob:.4f}"
                )

            # Return the token with the highest probability
            token_id = top_k_ids[0].item()
            return self.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        elif self.model_type == "encoder-decoder":
            # Generate encoder-decoder output
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                eos_token_id=8,
                max_new_tokens=amount_new_tokens,
            )
            output_text = self.tokenizer.decode(
                output_ids.squeeze(), skip_special_tokens=True
            )
            return output_text
        else:
            # Generate decoder output
            inputs.pop("token_type_ids", None)

            output_ids = self.model.generate(**inputs, max_new_tokens=amount_new_tokens)
            output_text = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=False
            )
            return output_text

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
            template_json=template_str,
        )
        print(f"Prompt:\n{prompt}")

        # Generate the filled JSON based on the prompt
        output_text = self.generate(prompt)
        print(f"Output text:\n{output_text}")

        filled_json = {}
        try:
            # Decoder models
            if self.model_type == "decoder":
                output_text = output_text.replace(self.tokenizer.eos_token, "")
                filled_json = json.loads(output_text.split(END_OF_PROMPT_MARKER)[-1])

            elif self.model_type == "encoder-decoder":
                # Attempt to parse the output text back into a JSON object.
                filled_json = json.loads(output_text)

            elif self.model_type == "encoder":
                start_index = output_text.find("[ {")
                end_index = output_text.find("} ]") + len("} ]")
                if start_index != -1 and end_index != -1:
                    output_text = output_text[start_index:end_index]

                filled_json = json.loads(output_text)

        except json.JSONDecodeError:
            print("Failed to parse model output into JSON. Raw output:", output_text)

        return filled_json
