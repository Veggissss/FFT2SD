from enum import Enum


class BaseEnum(Enum):
    """
    BaseEnum is a base class for creating enumerations.
    """

    @classmethod
    def get_enum_map(cls: Enum) -> dict[str, "BaseEnum"]:
        """
        Get the internal `_value2member_map_` attribute from the Enum class.
        """
        return cls._value2member_map_


class ModelType(BaseEnum):
    """
    Enum class to represent the type of model architecture.
    The strings are used for defining output dirs, in the models dict and for converting API str input to enum.
    """

    ENCODER = "encoder"
    DECODER = "decoder"
    ENCODER_DECODER = "encoder-decoder"


class ReportType(BaseEnum):
    """
    Enum class to represent the type of report.
    """

    KLINISK = "klinisk"
    MAKROSKOPISK = "makroskopisk"
    MIKROSKOPISK = "mikroskopisk"


class GenerationState(Enum):
    """
    Enum class to represent the state of the generation process.
    """

    WAITING = 0
    AWAITING_QUOTE = 1
    AWAITING_END_BRACKET = 2
