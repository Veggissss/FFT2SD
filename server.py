import copy
from flask import Flask, request, jsonify

from model_loader import ModelLoader
from utils.config import CONTAINER_NUMBER_MASK
from utils.enums import ModelType, ReportType
from utils.file_loader import load_json
from dataset_loader import reset_value_fields

app = Flask(__name__)

# Global variable to store the loaded model
model_loader = None
IS_TRAINED = True


def load_model(model_type: ModelType) -> str:
    """Function to load the specified LLM model based on type."""

    # Update the global model_loader variable
    global model_loader
    model_loader = ModelLoader(model_type, IS_TRAINED)

    return f"Loaded model: {model_loader.model_name} | {model_type}"


def fill_json(input_text: str, container_str: str, template_entry: dict) -> dict:
    """Function to fill a single JSON template using the loaded model."""
    mask_token = "null"
    if model_loader.model_type == ModelType.ENCODER:
        mask_token = model_loader.tokenizer.mask_token
    template_entry = reset_value_fields([template_entry], value=mask_token)[0]

    # Generate filled JSON using the model
    return model_loader.generate_filled_json(
        input_text, container_str, copy.deepcopy(template_entry)
    )


def generate(input_text: str) -> dict | None:
    """Function to generate text using the loaded model."""

    # Load metadata template to determine the report type and container count
    metadata_json = load_json("data_model/out/generated-metadata.json")
    report_json = metadata_json[0]
    glass_amount_json = metadata_json[1]

    # Find out the report type based on the input text
    report_type = fill_json(input_text, CONTAINER_NUMBER_MASK, report_json).get("value")
    if not report_type or report_type.strip() not in ReportType.get_enum_map():
        print("ERROR: Invalid report type!")
        return None

    # Find out the total number of containers based on the input text
    try:
        total_containers = int(
            fill_json(input_text, CONTAINER_NUMBER_MASK, glass_amount_json).get("value")
        )
        if total_containers < 1 or total_containers > 10:
            print("ERROR: Invalid container count!")
            return None
    except (ValueError, TypeError) as e:
        print(f"ERROR: Could not parse the container count! {e}")
        return None

    # Load the generated JSON template based on the report type
    template_json = load_json(f"data_model/out/generated-{report_type.strip()}.json")

    # Get the filled JSON for each container, 1 indexed
    final_json = {"input": input_text, "output": [], "report_type": report_type.strip()}
    for container_number in range(1, total_containers + 1):
        container_json = {f"glass {container_number}": []}
        for template_entry in template_json:
            # Generate filled JSON using the model
            filled_json = fill_json(input_text, container_number, template_entry)
            container_json[f"glass {container_number}"].append(filled_json)
        final_json["output"].append(container_json)

    return final_json


@app.route("/load_model", methods=["POST"])
def load_model_endpoint():
    """Endpoint to load the specified model based on type."""
    model_type_str: str | None = request.json.get("model_type")
    if not model_type_str:
        return (
            jsonify(
                {"error": "Model type is required (decoder, encoder, encoder-decoder)"}
            ),
            400,
        )
    if model_type_str not in ModelType.get_enum_map():
        print("Invalid model type")
        return jsonify({"error": "Invalid model type"}), 400

    return jsonify({"message": load_model(ModelType(model_type_str))})


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    """Endpoint to generate structured data as json based on text input."""
    if model_loader is None:
        return jsonify({"error": "Model is not loaded!"}), 400

    input_text: str | None = request.json.get("input_text")
    if not input_text:
        print("Input text is required")
        return jsonify({"error": "Input text is required"}), 400

    final_json = generate(input_text)
    if final_json is None:
        return jsonify({"error": "Failed to generate structured data"}), 500

    return jsonify(final_json)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
