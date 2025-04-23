from dataclasses import dataclass
from utils.enums import ModelType, ModelSize


@dataclass
class ModelSettings:
    """
    Model settings for the Hugging Face models. (This is not an enum I know...)
    """

    model_name: str
    use_peft: bool = False

    def get_saved_name(self) -> str:
        """
        Get the saved model name.
        """
        if self.use_peft:
            return self.model_name + "_peft"
        return self.model_name


# Definitions of Hugging Face models
MODELS_DICT: dict[ModelType, dict[ModelSize, ModelSettings]] = {
    ModelType.ENCODER_DECODER: {
        ModelSize.LARGE: ModelSettings("ltg/nort5-large"),
        ModelSize.BASE: ModelSettings("ltg/nort5-base"),
        ModelSize.SMALL: ModelSettings("ltg/nort5-small"),
    },
    ModelType.ENCODER: {
        ModelSize.LARGE: ModelSettings("ltg/norbert3-large"),
        ModelSize.BASE: ModelSettings("ltg/norbert3-base"),
        ModelSize.SMALL: ModelSettings("ltg/norbert3-small"),
    },
    ModelType.DECODER: {
        ModelSize.LARGE: ModelSettings("norallm/normistral-11b-warm"),
        ModelSize.BASE: ModelSettings("norallm/normistral-7b-warm"),
        ModelSize.SMALL: ModelSettings("norallm/normistral-7b-warm", use_peft=True),
    },
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
- container_number: The number of the container. Example: "Glass nummer 3"
- template_json: The JSON template to fill in.
"""
SYSTEM_PROMPT = (
    "Gitt teksten: {input_text}."
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
