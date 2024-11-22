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
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        config = BitsAndBytesConfig(load_in_4bit=True)

        # TODO: Load trained model from a custom local path!
        if model_type == "encoder-decoder":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(self.device)

        elif model_type == "decoder":
            # .to(self.device) is not used when using quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=config
            )
        else:
            model = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True
            )
            model.to(self.device)

        # Set special tokens for the tokenizer
        tokenizer.unk_token = "<unk>"
        tokenizer.pad_token = "<pad>"
        tokenizer.mask_token = "null"
        tokenizer.eos_token = "</s>"

        tokenizer.add_special_tokens(
            {
                "unk_token": tokenizer.unk_token,
                "pad_token": tokenizer.pad_token,
                "mask_token": tokenizer.mask_token,
                "eos_token": tokenizer.eos_token,
            }
        )

        # Resize the token embeddings to match the tokenizer
        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate model output based on the input prompt.
        :param prompt: Text input prompt for the model.
        :return: Generated output text.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

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
        else:
            # Generate decoder/encoder-decoder output
            output_ids = self.model.generate(**inputs, max_length=max_length)
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
        print(prompt)

        output_text = self.generate(prompt)

        try:
            # Decoder models
            if self.model_type == "decoder":
                output_text = output_text.replace(self.tokenizer.eos_token, "")
                filled_json = json.loads(output_text.split(END_OF_PROMPT_MARKER)[-1])

            elif self.model_type == "encoder-decoder":
                # TODO Improve: T5 have problem with json formatting, maybe add { and } to its tokenizer?
                is_alternate = False
                while self.tokenizer.unk_token in output_text:
                    symbol = "}" if is_alternate else "{"
                    output_text = output_text.replace(
                        self.tokenizer.unk_token, symbol, 1
                    )
                    is_alternate = not is_alternate

                output_text = output_text.replace(self.tokenizer.pad_token, "")
                output_text = output_text.replace(self.tokenizer.eos_token, "")

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
            filled_json = {}

        return filled_json
