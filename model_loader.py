import json
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

        # tokenizer.add_tokens(["\n"])

        # Resize the token embeddings to match the tokenizer
        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def generate(self, prompt: str, template_str: str) -> str:
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

        # Get the amount of new tokens to generate based on the json template
        amount_new_tokens = tokenized_template["input_ids"].shape[1]
        print(f"Max new decoder tokens: {amount_new_tokens}")

        if self.model_type == "encoder":
            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Find the masked token position
            masked_index = torch.where(
                inputs.input_ids == self.tokenizer.mask_token_id
            )[1].item()

            # Get logits for the masked token
            masked_token_logits = logits[0, masked_index]

            # Apply softmax to convert logits to probabilities
            probabilities = torch.nn.functional.softmax(masked_token_logits, dim=-1)

            # Filter the allowed tokens' probabilities based on template type
            print(f"Template: {template_str}")
            template_json = json.loads(template_str)
            template_type = template_json["type"]
            template_enums = None
            if template_type == "enum":
                template_enums = template_json["enum"]

            # Get allowed tokens
            allowed_token_ids = get_allowed_tokens(
                self.tokenizer, template_type, template_enums
            )

            # Check if allowed tokens are empty, and if so, use the model's output without filtering
            if len(allowed_token_ids) == 0:
                print(
                    "No allowed tokens found! Using the token with the highest probability instead."
                )
                allowed_token_ids = None  # Set to None to signal using all tokens

            allowed_token_probabilities = []
            # Get token probabilities for allowed tokens or use the highest probability if no allowed tokens
            if allowed_token_ids:
                allowed_token_probabilities = [
                    (token_id, probabilities[token_id].item())
                    for token_id in allowed_token_ids
                ]
            else:
                # If no allowed tokens are provided, select the top tokens based on probability
                top_tokens = probabilities.topk(10).indices
                allowed_token_probabilities = [
                    (token_id.item(), probabilities[token_id].item())
                    for token_id in top_tokens
                ]

            # Print top-k tokens with their probabilities
            top_k = sorted(
                allowed_token_probabilities, key=lambda x: x[1], reverse=True
            )[:10]
            print("Top tokens and their probabilities:")
            for token_id, prob in top_k:
                token = self.tokenizer.decode(token_id)
                print(f"Token: {token}, Probability: {prob:.4f}")

            # Select the allowed token with the highest probability (or the top one if no allowed tokens)
            predicted_token_id = max(allowed_token_probabilities, key=lambda x: x[1])[0]

            # Convert token ID back to token
            predicted_token = self.tokenizer.decode(predicted_token_id)

            # Replace the [MASK] token with the predicted token in the original sequence
            input_ids = inputs.input_ids[0].tolist()
            input_ids[masked_index] = predicted_token_id

            print(f"Predicted token: {predicted_token}")

            # Convert the updated input_ids back to text
            return self.tokenizer.decode(
                input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
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

        prompt += "\nUtfylt JSON verdi for 'value':\n{"

        # Generate the filled JSON based on the prompt
        output_text = self.generate(prompt, template_str)
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
                start_index = output_text.find("{")
                end_index = output_text.find("}") + len("}")

                # Replace single quotes with double quotes for JSON parsing
                output_text = output_text.replace("'", '"')

                if start_index != -1 and end_index != -1:
                    output_text = output_text[start_index:end_index]

                filled_json = json.loads(output_text)

        except json.JSONDecodeError:
            print("Failed to parse model output into JSON. Raw output:", output_text)

        return filled_json
