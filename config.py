from utils.enums import ModelType
from utils.data_classes import ModelSettings

# fmt: off
# Definitions of Hugging Face models
MODELS_DICT: dict[ModelType, list[ModelSettings]] = {
    ModelType.ENCODER_DECODER: [
        ModelSettings("ltg/nort5-small", training_batch_size=16, training_learning_rate=5e-4),
        ModelSettings("ltg/nort5-base", training_batch_size=8, training_learning_rate=3e-4),
        ModelSettings("ltg/nort5-large", training_batch_size=1, training_learning_rate=1e-4),
    ],
    ModelType.ENCODER: [
        #ModelSettings("ltg/norbert3-small", training_batch_size=16, training_learning_rate=5e-4, training_encoder_only_mask_values=True),
        ModelSettings("ltg/norbert3-small", training_batch_size=16, training_learning_rate=5e-4),
        ModelSettings("ltg/norbert3-base", training_batch_size=8, training_learning_rate=3e-4),
        ModelSettings("ltg/norbert3-large", training_batch_size=4, training_learning_rate=2e-4),
    ],
    ModelType.DECODER: [
        ModelSettings("norallm/normistral-7b-warm", training_batch_size=2, training_learning_rate=1e-4, use_4bit_quant=True),
        ModelSettings("norallm/normistral-7b-warm", training_batch_size=1, training_learning_rate=5e-5),
        ModelSettings("google/gemma-3-27b-it"),
        ModelSettings("Qwen/Qwen3-32B"),
    ],
}
# fmt: on

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
    "Finn kun informasjon som gjelder glass nummer {container_id}.\n"
    "Ignorer all informasjon om andre glass.\n"
    'Fyll ut feltet "value" basert på denne informasjonen.\n'
    'Hvis det ikke finnes en gyldig verdi, sett "value" til null.\n\n'
    f"{JSON_START_MARKER}\n"
    "{template_json}"
)

"""
Used for masking out "Glass nummer X." in the input prompt.
Only used for when asking for glass amount in training and inference.
"""
CONTAINER_ID_MASK = "?"
