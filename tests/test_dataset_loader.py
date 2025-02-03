import pytest
import dataset_loader


def test_dataset_loader():
    dataset, enums = dataset_loader.create_dataset(
        "data/test_data", "decoder", "[TEST_MASK]"
    )
    assert len(dataset["input"]) == len(dataset["output"])
    assert len(enums) > 0

    print(dataset["input"][0])


if __name__ == "__main__":
    pytest.main()
