from typing import Literal
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig,
)
import torch
import json


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

    def __load_model(self, model_name: str, model_type: str) -> tuple:
        """
        Load a model based on the specified type.
        :param model_name: The Hugging Face model name.
        :param model_type: Type of model - 'encoder', 'encoder-decoder', or 'decoder'.
        :return: model and tokenizer.
        """
        config = BitsAndBytesConfig(load_in_8bit=True)

        # TODO: Load trained model from a custom local path!
        if model_type == "encoder-decoder":
            # model = AutoModelForSeq2SeqLM.from_pretrained("/path/to/local")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(self.device)
        elif model_type == "decoder":
            # .to(self.device) is not used when using quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=config
            )
        else:
            model = AutoModel.from_pretrained(model_name)
            model.to(self.device)

        # Load corresponding tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            # padding=True
        ).to(self.device)

        if self.model_type == "encoder":
            output = self.model(**inputs)
            # TODO: Extract embeddings and map them to values
            print("TODO: Custom extraction needed for encoder models!")
            output_text = "WIP"
            return output_text
        else:
            # Generate decoder/encoder-decoder output
            output_ids = self.model.generate(**inputs, max_length=max_length)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output_text

    def generate_filled_json(self, input_text: str, json_template: dict) -> dict:
        """
        Fill out the JSON values based on the input text.
        :param model_loader: ModelLoader object with the loaded model and tokenizer.
        :param text: Input text to extract information from.
        :param json_template: JSON template with fields to be filled.
        :return: JSON with filled values.
        """

        # Convert JSON template to a string to include in the prompt.
        template_str = json.dumps(json_template, indent=2)
        prompt_separator = " END_OF_PROMPT "

        # TODO: Extract 'system prompt' generation to a separate function/class var. Must be consistent with the model's training data.
        prompt = f"'{input_text}'\nUse this information to replace the 'None' values in the json:\n{template_str}{prompt_separator}"
        # print(prompt)

        output_text = self.generate(prompt)

        # TODO T5 have problem with json formatting (add { and } to tokenizer or convert input?)

        try:
            # Attempt to parse the output text back into a JSON object.
            filled_json = json.loads(output_text.split(prompt_separator)[-1])
        except json.JSONDecodeError:
            print("Failed to parse model output into JSON. Raw output:", output_text)
            filled_json = {}

        return filled_json
