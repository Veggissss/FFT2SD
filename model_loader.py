from typing import Literal
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModel,
)
import json


class ModelLoader:
    def __init__(
        self,
        model_name: str,
        model_type: Literal["decoder", "encoder", "encoder-decoder"],
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.model, self.tokenizer = self.load_model(model_name, model_type)

    def load_model(self, model_name: str, model_type: str):
        """
        Load a model based on the specified type.
        :param model_name: The Hugging Face model name.
        :param model_type: Type of model - 'encoder', 'encoder-decoder', or 'decoder'.
        :return: model and tokenizer.
        """
        if model_type == "encoder-decoder":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif model_type == "decoder":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)

        # Load corresponding tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer

    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate model output based on the input prompt.
        :param prompt: Text input prompt for the model.
        :return: Generated output text.
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=max_length, truncation=True
        )

        if self.model_type == "encoder":
            output = self.model(**inputs)
            # TODO: Extract embeddings and map them to values
            print("TODO: Custom extraction needed for encoder models!")
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

        # TODO: Extract 'system prompt' generation to a separate function/class var.
        prompt = f"Extract information:\n{input_text}\nUse the information extracted from above to replace the 'value': null values in the json:\n{template_str}\n"
        # print(prompt)

        output_text = self.generate(prompt, max_length=1024)
        print(f"Model output:\n{output_text}")

        try:
            # Attempt to parse the output text back into a JSON object.
            filled_json = json.loads(output_text)
        except json.JSONDecodeError:
            print("Failed to parse model output into JSON. Raw output:", output_text)
            filled_json = {}

        return filled_json
