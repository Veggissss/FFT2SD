import pytest
import json
import dataset_loader


def test_dataset_loader_decoder():
    dataset, enums = dataset_loader.create_dataset("data/test_data", "decoder")
    assert len(dataset["input"]) == len(dataset["output"])
    assert len(enums) > 0

    # Check that the output column is not used
    assert dataset["output"][0] == "[UNUSED]"

    print(dataset["input"][0])
    print(dataset["output"][0])


def test_dataset_loader_encoder():
    dataset, enums = dataset_loader.create_dataset("data/test_data", "encoder")
    assert len(dataset["input"]) == len(dataset["output"])
    assert len(enums) > 0

    # Check that the output column is not used
    assert dataset["output"][0] == "[UNUSED]"

    print(dataset["input"][0])
    print(dataset["output"][0])


def test_dataset_loader_encoder_decoder():
    dataset, enums = dataset_loader.create_dataset("data/test_data", "encoder-decoder")
    assert len(dataset["input"]) == len(dataset["output"])
    assert len(enums) > 0

    # Confirm that the correct answer is not filled out
    assert "null" in dataset["input"][0]

    # Assert that the output is a json with a filled out value
    assert json.loads(dataset["output"][0]).get("value") is not None

    print(dataset["input"][0])
    print(dataset["output"][0])


if __name__ == "__main__":
    pytest.main()
