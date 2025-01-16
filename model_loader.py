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

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=q_config,
            ).to(self.device)
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
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # The prompt is always longer than the output
        amount_new_tokens = min(inputs["input_ids"].shape[1], max_length)
        print(f"Prompt tokenized length: {amount_new_tokens}")

        if self.model_type == "encoder":
            # Generate encoder output
            output = self.model(**inputs)

            # Replace masked tokens with the predicted tokens
            # mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
            mask_id = self.tokenizer.mask_token_id

            output_ids = torch.where(
                inputs.input_ids == mask_id, output.logits.argmax(-1), inputs.input_ids
            )

            return self.tokenizer.decode(
                output_ids[0].tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        elif self.model_type == "encoder-decoder":

            # Generate decoder/encoder-decoder output
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                eos_token_id=8,
                max_new_tokens=amount_new_tokens,
            )
            output_text = self.tokenizer.decode(
                output_ids.squeeze(), skip_special_tokens=False
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

    def generate_filled_json(self, input_text: str, template_str: str) -> dict:
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
            template_json=template_str,
        )
        print(f"Prompt:\n{prompt}")

        # Generate the filled JSON based on the prompt
        output_text = self.generate(prompt)

        filled_json = {}
        try:
            # Decoder models
            if self.model_type == "decoder":
                output_text = output_text.replace(self.tokenizer.eos_token, "")
                filled_json = json.loads(output_text.split(END_OF_PROMPT_MARKER)[-1])

            elif self.model_type == "encoder-decoder":
                # Find the start and end of the tokens
                start_token = "[BOS]"
                end_token = "[MASK_"

                # Get the indices of the tokens
                start_index = output_text.find(start_token) + len(start_token)
                end_index = output_text.find(end_token)

                # Extract the substring
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    output_text = output_text[
                        start_index:end_index
                    ].strip()  # Extract and strip whitespace
                else:
                    print(
                        f"Failed to extract substring from model output: {output_text}"
                    )

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
