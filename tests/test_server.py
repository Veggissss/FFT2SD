import pytest
from flask.testing import FlaskClient
import server


@pytest.fixture
def client() -> FlaskClient:
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
    response = client.post("/generate", json={"input_text": "Sample input text"})
    assert response.status_code == 400
    assert "Model is not loaded!" in response.json["error"]

    # Load the model for testing
    client.post("/load_model", json={"model_type": "encoder"})

    response = client.post("/generate", json={"input_text": "Sample input text"})
    assert response.status_code == 200
    assert "input" in response.json
    assert "output" in response.json
    assert "report_type" in response.json

    response = client.post("/generate", json={})
    assert response.status_code == 400
    assert "Input text is required" in response.json["error"]
