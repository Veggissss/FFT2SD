import json
import torch
from token_constraints import get_allowed_tokens, StopOnToken
from config import END_OF_PROMPT_MARKER


class BaseGenerationStrategy:
    """
    Base class for generation strategies. Every model type (encoder, decoder, encoder-decoder) will implement this.
    """

    def generate(
        self,
        model_loader,
        inputs: dict[str, torch.Tensor],
        amount_new_tokens: int,
        template_str: str = None,
        stopping_criteria: StopOnToken = None,
    ) -> str:
        raise NotImplementedError

    def output_to_json(self, output_text: str) -> dict:
        raise NotImplementedError


class EncoderDecoderStrategy(BaseGenerationStrategy):

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        template_str=None,
        stopping_criteria=None,
    ) -> str:
        output_ids = model_loader.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            eos_token_id=8,
            max_new_tokens=amount_new_tokens,
            stopping_criteria=stopping_criteria,
        )
        return model_loader.tokenizer.decode(
            output_ids.squeeze(), skip_special_tokens=True
        )

    def output_to_json(self, output_text: str) -> dict:
        return json.loads(output_text)


class DecoderStrategy(BaseGenerationStrategy):

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        template_str=None,
        stopping_criteria=None,
    ) -> str:
        # Generate decoder output
        inputs.pop("token_type_ids", None)

        output_ids = model_loader.model.generate(
            **inputs,
            max_new_tokens=amount_new_tokens,
            stopping_criteria=stopping_criteria,
        )
        return model_loader.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=False,
        )

    def output_to_json(self, output_text: str) -> dict:
        print(output_text)
        # output_text = output_text.replace(model_loader.tokenizer.eos_token, "")
        return json.loads(output_text.split(END_OF_PROMPT_MARKER)[-1])


class EncoderStrategy(BaseGenerationStrategy):

    def generate(
        self,
        model_loader,
        inputs,
        amount_new_tokens,
        template_str=None,
        stopping_criteria=None,
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
        template_json = json.loads(template_str)
        template_type = template_json["type"]

        # If the template is an enum, get the allowed tokens
        template_enums = template_json.get("enum", None)

        # Get allowed tokens
        allowed_token_ids = get_allowed_tokens(
            model_loader.tokenizer, template_type, template_enums
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

    def output_to_json(self, output_text: str) -> dict:
        start_index = output_text.find("{")
        end_index = output_text.find("}") + len("}")

        # Extract the JSON part from the output text
        if start_index != -1 and end_index != -1:
            output_text = output_text[start_index:end_index]

        return json.loads(output_text)
