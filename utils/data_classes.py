import copy
from dataclasses import dataclass
from utils.enums import ReportType


@dataclass
class TemplateGeneration:
    """
    Data class object that holds the input information for the model generation.

    input_text: The raw prompt text to be used for generation.
    container_id: The identifier of the container being processed.
    template_json: List of JSON template dictionaries with fields to be filled.
    original_template_json: A copy of the original template JSON for reference.
    """

    input_text: str
    container_id: str
    template_json: list[dict]
    original_template_json: list[dict] | None = None

    def copy_template(self) -> None:
        """
        Create a copy of the template JSON to avoid modifying the original.
        """
        self.original_template_json = copy.deepcopy(self.template_json)


@dataclass()
class TokenOptions:
    """
    Optional token options for model generation.

    report_type: Report type used for caching of allowed tokens.
    allow_null: Whether to allow null values in the output.
    reduce_null_bias: Reduce the chance of generating null values by subtracting the value from the logits.
    include_enums: Whether to include enum values in the prompt. (Won't affect the encoder type)
    generate_strings: Whether to generate string values in the output, if False string values will be 'null'.
    """

    report_type: ReportType | None = None
    allow_null: bool = True
    reduce_null_bias: float = 0.0
    include_enums: bool = False
    generate_strings: bool = False


@dataclass(frozen=True)
class ModelSettings:
    """
    Model specific settings for the Hugging Face models.
    """

    model_name: str
    use_peft: bool = False

    def __str__(self) -> str:
        """
        Get the saved model name.
        """
        if self.use_peft:
            return self.model_name + "_peft"
        return self.model_name
