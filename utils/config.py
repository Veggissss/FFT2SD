# Definitions of Hugging Face models
MODELS_DICT: dict[str, str] = {
    "encoder-decoder": "ltg/nort5-large",
    "decoder": "norallm/normistral-7b-warm",
    "encoder": "ltg/norbert3-large",
}
DATA_MODEL_OUTPUT_FOLDER = "data_model/out"

DEBUG_MODE_ENABLED = True
REDUCE_NULL_BIAS = 2.0

"""
Mark the end of the model prompt and before the template JSON in the prompt.
Useful for splitting json from the final output from decoder and encoder models.
"""
JSON_START_MARKER = "<JSON_START>"

"""
The prompt for the system to ask the model to fill in the missing JSON values.

Parameters:
- input_text: The input text to extract information from.
- container_number: The number of the container. Example: "Glass nummer 3"
- template_json: The JSON template to fill in.
"""
SYSTEM_PROMPT = (
    "Gitt teksten: '{input_text}'."
    + "\nGlass nummer {container_number}."
    + '\nFyll ut den manglede verdien for JSON feltet "value": '
    + JSON_START_MARKER
    + "\n{template_json}"
)

"""
Used for masking out "Glass nummer X." in the input prompt.
Only used for when asking for glass amount in training and inference.
"""
CONTAINER_NUMBER_MASK = "?"
