import pytest
import json
from dataset_loader import DatasetLoader
from utils.enums import ModelType
from config import JSON_START_MARKER

# Test data path
DATA_PATH = "data/test_data"


def test_dataset_loader_decoder():
    dataset, enums = DatasetLoader(ModelType.DECODER).create_dataset(
        DATA_PATH, include_enums=True
    )
    assert len(dataset["input"]) == len(dataset["output"])
    assert len(enums) > 0

    # Check that the enums contain the correct values for null and not None
    assert not "None" in enums
    assert "null" in enums

    # Check that the output column is not used
    assert dataset["output"][0] == "[UNUSED BY THE DECODER TYPE]"

    # Check that enum values are in the prompt
    assert '"type": "enum", "enum"' in dataset["input"][0]

    print(dataset["input"][0])
    print(dataset["output"][0])


def test_dataset_loader_encoder():
    dataset, enums = DatasetLoader(
        ModelType.ENCODER, mask="[TEST_MASK]"
    ).create_dataset(DATA_PATH, False)
    assert len(dataset["input"]) == len(dataset["output"])
    assert len(enums) > 0

    # Check that the enums contain the correct values for null and not None
    assert not "None" in enums
    assert "null" in enums

    # Check that the output column is used for masking value position
    assert "[TEST_MASK]" in dataset["output"][0]

    # Check that enum values are NOT in the prompt
    assert '"type": "enum", "enum"' not in dataset["input"][0]

    print(dataset["input"][0])
    print(dataset["output"][0])


def test_dataset_loader_encoder_decoder():
    dataset, enums = DatasetLoader(ModelType.ENCODER_DECODER).create_dataset(
        DATA_PATH, False
    )
    assert len(dataset["input"]) == len(dataset["output"])
    assert len(enums) > 0

    # Check that the enums contain the correct values for null and not None
    assert not "None" in enums
    assert "null" in enums

    # Confirm that the correct answer is not filled out
    print(dataset["input"][0])
    assert "null" in dataset["input"][0]

    # Check that enum values are NOT in the prompt
    assert '"type": "enum", "enum"' not in dataset["input"][0]

    # Assert that the output is a json with a filled out value
    assert json.loads(dataset["output"][0]).get("value") is not None

    print(dataset["input"][0])
    print(dataset["output"][0])


def test_process_enum_file():
    # Test with empty dataset path
    dataset_path = "tests/temp/empty"
    dataset_loader = DatasetLoader(ModelType.ENCODER)
    enum_dataset, enums = dataset_loader.create_dataset(
        dataset_path, include_enums=False
    )
    assert len(enums) == 0
    assert len(enum_dataset["input"]) != 0
    assert len(enum_dataset["output"]) != 0

    assert len(enum_dataset["output"]) == len(enum_dataset["input"])

    # Should use enum values as correct values
    assert not "null" in str(enum_dataset["input"][0]).split(JSON_START_MARKER)[1]
    assert "null" in str(enum_dataset["output"][0]).split(JSON_START_MARKER)[1]


if __name__ == "__main__":
    pytest.main()
