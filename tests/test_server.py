import os
from typing import Generator
import pytest
from flask.testing import FlaskClient
import server
from utils.file_loader import save_json, load_json


@pytest.fixture
def client() -> Generator[FlaskClient, None, None]:
    server.app.config["TESTING"] = True
    with server.app.test_client() as client:
        yield client


def test_load_model_endpoint(client: FlaskClient):
    response = client.post("/load_model", json={"model_type": "encoder"})
    assert response.status_code == 200
    assert "Loaded model:" in response.json["message"]

    response = client.post("/load_model", json={"model_type": "invalid"})
    assert response.status_code == 400
    assert "Invalid model type" in response.json["error"]

    response = client.post("/load_model", json={})
    assert response.status_code == 400
    assert "Model type is required " in response.json["error"]


def test_generate_endpoint(client: FlaskClient):
    # Test without loading model
    server.model_loader = None
    response = client.post("/generate", json={"input_text": "text"})
    assert response.status_code == 400
    assert "Model is not loaded!" in response.json["error"]

    # Load the model for testing
    client.post("/load_model", json={"model_type": "encoder"})

    response = client.post(
        "/generate",
        json={
            "input_text": "3 glass, merket 1 - 3\n1: 4  gryn i #1\n2: 3  gryn i #2\n3: 7  gryn i #3"
        },
    )
    assert response.status_code == 200

    # Get first json object from the response
    assert isinstance(response.json, list)
    assert len(response.json) > 0
    json_response = response.json[0]

    assert "input_text" in json_response
    assert "metadata_json" in json_response
    assert "target_json" in json_response

    # Test with invalid input
    response = client.post("/generate", json={})
    assert response.status_code == 400
    assert "Input text is required" in response.json["error"]


def test_unlabeled_endpoint(client: FlaskClient):
    # Test getting an unlabeled case
    response = client.get("/unlabeled/klinisk")
    assert response.status_code == 200
    assert "id" in response.json
    assert "text" in response.json
    assert not response.json["is_diagnose"]

    # Test with diagnose type
    response = client.get("/unlabeled/diagnose")
    assert "id" in response.json
    assert "text" in response.json
    assert response.json["is_diagnose"]

    # Test with undefined type
    response = client.get("/unlabeled/null")
    assert "id" in response.json
    assert "text" in response.json
    assert not response.json["is_diagnose"]


def test_correct_endpoint(client: FlaskClient):
    test_path = "./tests/temp/"
    labeled_path = f"{test_path}labeled_ids.json"
    test_label_path = f"{test_path}klinisk_TEST_ID_1.json"

    if os.path.exists(labeled_path):
        os.remove(labeled_path)

    if os.path.exists(test_label_path):
        os.remove(test_label_path)

    # Create test labeled_ids.json file
    empty_json = {}
    save_json(empty_json, labeled_path)

    server.CORRECTED_OUT_DIR = test_path
    server.LABELED_IDS_PATH = labeled_path
    report_id = "TEST_ID"

    # Test using invalid report_type (test_type)
    response = client.post(
        f"/correct/{report_id}",
        json=[
            {
                "input_text": "text",
                "metadata_json": [{"value": "test_type"}, {"value": 1}],
                "target_json": [{"key": "value"}],
            }
        ],
    )
    assert response.status_code == 400
    assert response.json["error"] == "Invalid report type"

    # Test correcting without reports
    response = client.post(f"/correct/{report_id}", json=[])
    assert response.status_code == 400
    assert "No reports provided" in response.json["error"]

    # Test using valid report_type (klinisk)
    response = client.post(
        f"/correct/{report_id}",
        json=[
            {
                "input_text": "text",
                "metadata_json": [{"value": "klinisk"}, {"value": 1}],
                "target_json": [{"key": "value"}],
            }
        ],
    )
    assert response.status_code == 200
    assert "Correctly labeled JSON saved!" in response.json["message"]

    assert os.path.exists(test_label_path)

    labeled_ids = load_json(labeled_path)
    assert report_id in labeled_ids
    assert "kliniske_opplysninger" in labeled_ids[report_id]
    assert labeled_ids[report_id]["kliniske_opplysninger"]
