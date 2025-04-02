import uuid
import copy
from flask import Flask, request, jsonify
from flask_cors import CORS

from model_loader import ModelLoader
from utils.config import CONTAINER_NUMBER_MASK
from utils.enums import ModelType, ReportType
from utils.file_loader import load_json, save_json
from dataset_loader import reset_value_fields

app = Flask(__name__)

# Enable CORS for all endpoints
CORS(app, origins=["http://localhost:*", "http://127.0.0.1:*"])

# Global variable to store the loaded model
model_loader = None
# If to use the trained model or directly from huggingface specified in utils/config.py
IS_TRAINED = True

# Unlabeled dataset path
UNLABELED_BATCH_PATH = "data/large_batch/export_2025-03-17.json"
LABELED_IDS_PATH = "data/large_batch/labeled_ids.json"


def load_model(model_type: ModelType) -> str:
    """Function to load the specified LLM model based on type."""

    # Update the global model_loader variable
    global model_loader
    model_loader = ModelLoader(model_type, IS_TRAINED)
    model_loader.model.eval()

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
    report_type = None
    if report_type_str and report_type_str in ReportType.get_enum_map():
        report_type = ReportType(report_type_str)
    total_containers: int | None = request.json.get("total_containers")

    reports = generate(input_text, report_type, total_containers)
    if reports is None:
        return jsonify({"error": "Failed to generate structured data"}), 500

    return jsonify(reports)


@app.route("/correct/<string:report_id>", methods=["POST"])
def correct_endpoint(report_id: str):
    """Endpoint to save the corrected JSON by the user."""
    reports: list[dict] = request.json
    if not reports:
        return jsonify({"error": "No reports provided"}), 400

    if not report_id or report_id == "null":
        # Generate a new report ID if not provided
        report_id = str(uuid.uuid4())

    # Check if the report is already labeled
    labeled_ids_json: dict = load_json(LABELED_IDS_PATH)

    # Save every JSON in the list as a separate file
    for count, report in enumerate(reports):
        report_type = report["metadata_json"][0]["value"]

        # Verify if the report type is valid
        if report_type not in ReportType.get_enum_map():
            return jsonify({"error": "Invalid report type"}), 400

        save_json(report, f"data/corrected/{report_type}_{report_id}_{count+1}.json")
        labeled_ids_json.setdefault(report_id, {})[report_type] = True

    # Update the labeled IDs JSON
    save_json(labeled_ids_json, LABELED_IDS_PATH)

    return jsonify({"message": "Correctly labeled JSON saved!"})


@app.route("/unlabeled/<string:report_type_str>", methods=["GET"])
def unlabeled_endpoint(report_type_str: str):
    """Endpoint to get the unlabeled JSON files."""
    unlabeled_batch_json: list[dict] = load_json(UNLABELED_BATCH_PATH)
    labeled_ids_json: dict = load_json(LABELED_IDS_PATH)

    report_type = None
    if report_type_str and report_type_str in ReportType.get_enum_map():
        report_type = ReportType(report_type_str)

    for dataset_case in unlabeled_batch_json:
        if dataset_case["id"] in labeled_ids_json:
            is_klinisk_labeled = labeled_ids_json[dataset_case["id"]].get(
                ReportType.KLINISK.value, False
            )
            is_makroskopisk_labeled = labeled_ids_json[dataset_case["id"]].get(
                ReportType.MAKROSKOPISK.value, False
            )
            is_mikroskopisk_labeled = labeled_ids_json[dataset_case["id"]].get(
                ReportType.MIKROSKOPISK.value, False
            )
            if (
                is_klinisk_labeled
                and is_makroskopisk_labeled
                and is_mikroskopisk_labeled
            ):
                # Ignore the cases where all report types for the case are labeled
                continue

            if not report_type:
                # Get next report type for the case id if is not provided
                if not is_klinisk_labeled:
                    report_type = ReportType.KLINISK
                elif not is_makroskopisk_labeled:
                    report_type = ReportType.MAKROSKOPISK
                else:
                    report_type = ReportType.MIKROSKOPISK
            elif labeled_ids_json[dataset_case["id"]].get(report_type.value, False):
                # Ignore the cases where the report type for the case is labeled
                continue

        if not report_type:
            report_type = ReportType.KLINISK

        # Get the report text based on the report type using match statement
        match report_type:
            case ReportType.KLINISK:
                report_text = dataset_case["kliniske_opplysninger"]
            case ReportType.MAKROSKOPISK:
                report_text = dataset_case["makrobeskrivelse"]
            case ReportType.MIKROSKOPISK:
                report_text = dataset_case["mikrobeskrivelse"]
                if dataset_case["diagnose"] is not None:
                    report_text += "\n\n" + dataset_case["diagnose"]

        # Format the report text
        report_text = report_text.strip().replace("\r", "\n")

        return jsonify(
            {
                "id": dataset_case["id"],
                "report_type": report_type.value,
                "text": report_text,
            }
        )

    return jsonify({"error": "No unlabeled cases found!"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
