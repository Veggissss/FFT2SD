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
    reduce_null_bias: Set a value between 0 and 1 to reduce the bias towards null values. If the value is set to 0.8, if the confidence is below 80% the chance will be reduced by 80% of the null value.
    include_enums: Whether to include possible enum values in the prompt, increasing prompt token amount.
    generate_strings: Whether to generate string values in the output, if False string values will be 'null'. (Does not affect the encoder models)
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

    :param base_model_name: The name/repo path to the HuggingFace model.
    :param peft_model_name: The HuggingFace PEFT model repo path. If None, the local trained peft will be used if is_trained is True.
    :param use_4bit_quant: Whether to load the model using 4-bit quantization.
    :param is_fine_tuning: Whether the model is going to be fine-tuned or not.

    :param training_batch_size: The batch size to be used during training.
    :param training_num_epochs: The number of epochs to train the model.
    :param training_learning_rate: The learning rate to be used during training.
    :param training_encoder_only_mask_values: Whether to ONLY mask out the values when training encoder models. If false, use random mlm.
    """

    base_model_name: str
    peft_model_name: None | str = None
    use_4bit_quant: bool = False
    is_fine_tuning: bool = True

    training_batch_size: int = 1
    training_num_epochs: int = 20
    training_learning_rate: float = 1e-4
    training_encoder_only_mask_values: bool = False

    def __str__(self) -> str:
        """
        Get the saved model name.
        """
        if self.peft_model_name:
            return self.peft_model_name
        if self.training_encoder_only_mask_values:
            return self.base_model_name + "_mask_values"
        if self.use_4bit_quant:
            return self.base_model_name + "_4bit_quant"
        return self.base_model_name

    def __post_init__(self) -> None:
        """
        Post-initialization method to validate the model settings.
        """
        if self.training_batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        if self.training_num_epochs < 1:
            raise ValueError("Number of epochs must be at least 1.")
        if self.training_learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        if self.is_fine_tuning and self.peft_model_name:
            raise ValueError("PEFT model repo name should not be set when fine-tuning.")
