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
    ENCODER_DECODER = "encoder_decoder"
    DECODER = "decoder"


class ReportType(BaseEnum):
    """
    Enum class to represent the type of report.
    """

    KLINISK = "klinisk"
    MAKROSKOPISK = "makroskopisk"
    MIKROSKOPISK = "mikroskopisk"


class DatasetField(BaseEnum):
    """
    Enum class to represent the field names for the text types in the dataset.
    """

    KLINISK = "kliniske_opplysninger"
    MAKROSKOPISK = "makrobeskrivelse"
    MIKROSKOPISK = "mikrobeskrivelse"
    DIAGNOSE = "diagnose"


class GenerationState(Enum):
    """
    Enum class to represent the states of the restricted token generation.
    """

    WAITING = 0
    AWAIT_VALUE = 1
    AWAIT_BRACKET_END = 2
