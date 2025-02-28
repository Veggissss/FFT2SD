import pytest
import json
import dataset_loader
from utils.enums import ModelType

# Test data path
DATA_PATH = "data/test_data"


def test_dataset_loader_decoder():
    dataset, enums = dataset_loader.create_dataset(DATA_PATH, ModelType.DECODER)
    assert len(dataset["input"]) == len(dataset["output"])
    assert len(enums) > 0

    # Check that the enums contain the correct values for null and not None
    assert not "None" in enums
    assert "null" in enums

    # Check that the output column is not used
    assert dataset["output"][0] == "[UNUSED BY THE DECODER TYPE]"

    print(dataset["input"][0])
    print(dataset["output"][0])


def test_dataset_loader_encoder():
    dataset, enums = dataset_loader.create_dataset(DATA_PATH, ModelType.ENCODER)
    assert len(dataset["input"]) == len(dataset["output"])
    assert len(enums) > 0

    # Check that the enums contain the correct values for null and not None
    assert not "None" in enums
    assert "null" in enums

    # Check that the output column is not used
    assert dataset["output"][0] == "[UNUSED BY THE ENCODER TYPE]"

    print(dataset["input"][0])
    print(dataset["output"][0])


def test_dataset_loader_encoder_decoder():
    dataset, enums = dataset_loader.create_dataset(DATA_PATH, ModelType.ENCODER_DECODER)
    assert len(dataset["input"]) == len(dataset["output"])
    assert len(enums) > 0

    # Check that the enums contain the correct values for null and not None
    assert not "None" in enums
    assert "null" in enums

    # Confirm that the correct answer is not filled out
    print(dataset["input"][0])
    assert "null" in dataset["input"][0]

    # Assert that the output is a json with a filled out value
    assert json.loads(dataset["output"][0]).get("value") is not None

    print(dataset["input"][0])
    print(dataset["output"][0])


if __name__ == "__main__":
    pytest.main()
