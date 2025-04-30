from utils.enums import ModelType
from utils.data_classes import ModelSettings

# Definitions of Hugging Face models
MODELS_DICT: dict[ModelType, list[ModelSettings]] = {
    ModelType.ENCODER_DECODER: [
        ModelSettings("ltg/nort5-small"),
        ModelSettings("ltg/nort5-base"),
        ModelSettings("ltg/nort5-large"),
    ],
    ModelType.ENCODER: [
        ModelSettings("ltg/norbert3-small"),
        ModelSettings("ltg/norbert3-base"),
        ModelSettings("ltg/norbert3-large"),
    ],
    ModelType.DECODER: [
        ModelSettings("norallm/normistral-7b-warm", use_4bit_quant=True),
        ModelSettings("norallm/normistral-7b-warm"),
        ModelSettings("google/gemma-3-27b-it"),
        ModelSettings("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
    ],
}

# Output path for the data models combined struct/template
DATA_MODEL_OUTPUT_FOLDER = "data_model/out"

# Print out contstrained token probabilities and other debug information
DEBUG_MODE_ENABLED = True  # TODO: Add propper logging

"""
Mark the end of the model prompt and before the template JSON in the prompt.
Useful for splitting json from the final output from decoder and encoder models.
"""
JSON_START_MARKER = "<JSON_START>"

"""
The prompt for the system to ask the model to fill in the missing JSON values.

Parameters:
- input_text: The input text to extract information from.
- container_id: The number of the container. Example: "Glass nummer 3"
- template_json: The JSON template to fill in.
"""
SYSTEM_PROMPT = (
    "Gitt teksten:\n{input_text}\n\n"
    'Fyll ut feltet "value" basert på beskrivelsen av glass nummer {container_id}.\n'
    'Hvis det ikke finnes en gyldig verdi, sett "value" til null.\n'
    f"{JSON_START_MARKER}\n"
    "{template_json}"
)

"""
Used for masking out "Glass nummer X." in the input prompt.
Only used for when asking for glass amount in training and inference.
"""
CONTAINER_ID_MASK = "?"
