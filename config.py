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
        ModelSettings("norallm/normistral-7b-warm", use_peft=True),
        ModelSettings("norallm/normistral-7b-warm"),
        ModelSettings("norallm/normistral-11b-warm"),
        ModelSettings("google/gemma-3-27b-it"),
        ModelSettings("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
    ],
}

# Output path for the data models combined struct/template
DATA_MODEL_OUTPUT_FOLDER = "data_model/out"

# Print out contstrained token probabilities
DEBUG_MODE_ENABLED = True

# Whether the input prompt and training data should include the enum definitions or not.
# This uses a lot of tokens, but gives more context, specifically to untrained models.
INCLUDE_ENUMS = False

# Reduce the chance of null values in the output
REDUCE_NULL_BIAS = 0

"""
Whether or not to generate unrestricted string output then the "type" is set to "string".
This speeds up the generation process, but all the "value" for the strings will be set to null.
Used for fast eval metrics generation.
"""
STRING_GENERATION_ENABLED = True

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
    "Gitt teksten: {input_text}."
    + "\nGlass nummer {container_id}."
    + '\nFyll ut den manglede verdien for JSON feltet "value": '
    + JSON_START_MARKER
    + "\n{template_json}"
)

"""
Used for masking out "Glass nummer X." in the input prompt.
Only used for when asking for glass amount in training and inference.
"""
CONTAINER_ID_MASK = "?"
