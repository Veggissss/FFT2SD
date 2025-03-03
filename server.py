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


def generate(
    input_text: str, report_type: ReportType = None, total_containers: int = None
) -> list[dict] | None:
    """Function to generate text using the loaded model."""

    # Load metadata template to determine the report type and container count
    metadata_json = load_json("data_model/out/generated-metadata.json")
    report_json = metadata_json[0]
    glass_amount_json = metadata_json[1]
    container_id_json = metadata_json[2]

    # Determine report type if not provided
    if not report_type:
        filled_report = fill_json(input_text, CONTAINER_NUMBER_MASK, report_json)
        report_type_str = filled_report.get("value", "").strip()
        if not report_type_str or report_type_str not in ReportType.get_enum_map():
            print("ERROR: Invalid report type!")
            return None
        report_type = ReportType(report_type_str)

    # Determine total containers if not provided
    if not total_containers:
        filled_container = fill_json(
            input_text, CONTAINER_NUMBER_MASK, glass_amount_json
        )
        total_containers = filled_container.get("value")
        if not total_containers or not str(total_containers).isdigit():
            print("ERROR: Could not parse the container count!")
            return None
        total_containers = int(total_containers)
        if total_containers < 1 or total_containers > 10:
            print("ERROR: container count out of range!")
            return None

    # Load the generated JSON template based on the report type
    template_json = load_json(f"data_model/out/generated-{report_type.value}.json")

    # Get the filled JSON for each container, 1 indexed
    reports = []
    for container_number in range(1, total_containers + 1):
        glass_amount_json["value"] = total_containers
        report_json["value"] = report_type.value
        container_id_json["value"] = container_number
        generated_report = {
            "input_text": input_text,
            "target_json": [],
            "metadata_json": copy.deepcopy(metadata_json),
        }

        # Generate filled JSON using the model
        for template_entry in template_json:
            filled_json = fill_json(input_text, container_number, template_entry)
            generated_report["target_json"].append(filled_json)

        reports.append(generated_report)

    return reports


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

    # Get optional parameters
    report_type_str: str | None = request.json.get("report_type")
    report_type = ReportType(report_type_str) if report_type_str else None
    total_containers: int | None = request.json.get("total_containers")

    final_json = generate(input_text, report_type, total_containers)
    if final_json is None:
        return jsonify({"error": "Failed to generate structured data"}), 500

    return jsonify(final_json)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
