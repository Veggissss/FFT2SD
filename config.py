# Definitions of different models
MODELS_DICT: dict[str, str] = {
    # Trained local models
    "trained-encoder-decoder": "trained/encoder-decoder",
    "trained-decoder": "trained/decoder",
    "trained-encoder": "trained/encoder",
    # Hugging Face models
    "encoder-decoder": "ltg/nort5-small",
    "decoder": "norallm/normistral-7b-warm",
    "encoder": "ltg/norbert3-base",
}

"""
Mark the end of the prompt for the model to start generating the output.
Useful for decoder models which continue generating using the prompt.
"""
END_OF_PROMPT_MARKER = "[JSON_START]"

"""
The prompt for the system to ask the model to fill in the missing JSON values.

Parameters:
- input_text: The input text to extract information from.
- container_number: The number of the container. Example: "Glass nummer 3"
- template_json: The JSON template to fill in.
- decoder_start: The start of the prompt for decoder models. Empty for other models.
"""
SYSTEM_PROMPT = (
    "Gitt teksten: \n'{input_text}'.\nGlass nummer {container_number}.\nFyll ut den manglede verdien for feltet \"value\". Behold JSON strukturen: "
    + END_OF_PROMPT_MARKER
    + " {template_json}"
)
