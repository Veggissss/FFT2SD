# Definitions of Hugging Face models
MODELS_DICT: dict[str, str] = {
    "encoder-decoder": "ltg/nort5-small",
    "decoder": "norallm/normistral-7b-warm",
    "encoder": "ltg/norbert3-base",
}

"""
Mark the end of the model prompt and before the template JSON in the prompt.
Useful for splitting json from the final output from decoder and encoder models.
"""
END_OF_PROMPT_MARKER = "[JSON_START]"

"""
The prompt for the system to ask the model to fill in the missing JSON values.

Parameters:
- input_text: The input text to extract information from.
- container_number: The number of the container. Example: "Glass nummer 3"
- template_json: The JSON template to fill in.
"""
SYSTEM_PROMPT = (
    "Gitt teksten: \n'{input_text}'.\nGlass nummer {container_number}.\nFyll ut den manglede verdien for feltet \"value\". Behold JSON strukturen: "
    + END_OF_PROMPT_MARKER
    + " {template_json}"
)

"""
Used for masking out "Glass nummer X" in the input prompt.
Only used for when asking for glass amount in training and inference.
"""
CONTAINER_NUMBER_MASK = "?"
