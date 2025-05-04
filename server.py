import uuid
import os
import copy
from flask import Flask, request, jsonify
from flask_cors import CORS

from model_loader import ModelLoader
from config import CONTAINER_ID_MASK, MODELS_DICT
from utils.enums import ModelType, ReportType, DatasetField
from utils.data_classes import TemplateGeneration, TokenOptions
from utils.file_loader import load_json, save_json
from dataset_loader import reset_value_fields

app = Flask(__name__)

# Enable CORS for all endpoints
CORS(app, origins="*")

# Global variable to store the loaded model
model_loader: ModelLoader = None

# Unlabeled dataset path
UNLABELED_BATCH_PATH = "data/large_batch/export_2025-03-17.json"
LABELED_IDS_PATH = "data/large_batch/labeled_ids.json"
CORRECTED_OUT_DIR = "data/corrected/"


def load_model(model_type: ModelType, model_index: int, is_trained: bool) -> str:
    """Function to load the specified LLM model based on type."""
    # Update the global model_loader variable
    global model_loader
    if model_loader:
        model_loader.unload_model()

    model_loader = ModelLoader(model_type, model_index, is_trained)
    model_loader.model.eval()

    return f"Loaded model: {model_loader.model_name} | {model_type}"


def fill_json(
    generation: TemplateGeneration,
    token_options: TokenOptions = None,
) -> list[dict]:
    """Function to fill a single JSON template using the loaded model."""
    mask_token = "null"
    if model_loader.model_type == ModelType.ENCODER:
        mask_token = model_loader.tokenizer.mask_token
    generation.template_json = reset_value_fields(
        generation.template_json, value=mask_token
    )
    # Reset original template with masked token
    generation.copy_template()

    # Generate filled JSON using the model
    return model_loader.generate_filled_json(generation, token_options)


def generate(
    input_text: str,
    report_type: ReportType = None,
    total_containers: int = None,
    token_options: TokenOptions = None,
) -> list[dict] | None:
    """Function to generate structured data using the loaded model."""

    # Load metadata template to determine the report type and container count
    metadata_json = load_json("data_model/out/generated-metadata.json")
    report_json = metadata_json[0]
    glass_amount_json = metadata_json[1]
    container_id_json = metadata_json[2]

    # Determine report type if not provided
    if not report_type:
        filled_report = fill_json(
            TemplateGeneration(input_text, CONTAINER_ID_MASK, [report_json]),
            TokenOptions(allow_null=False),
        )[0]
        report_type_str = filled_report.get("value", "")
        if (
            not report_type_str
            or report_type_str.strip() not in ReportType.get_enum_map()
        ):
            print("ERROR: Invalid report type!")
            return None
        report_type = ReportType(report_type_str.strip())

    # Determine total containers if not provided
    if not total_containers:
        filled_container = fill_json(
            TemplateGeneration(input_text, CONTAINER_ID_MASK, [glass_amount_json]),
            TokenOptions(allow_null=False),
        )[0]
        total_containers = filled_container.get("value")
        if not total_containers or not str(total_containers).isdigit():
            print("ERROR: Could not parse the container count!")
            return None
        total_containers = int(total_containers)
        if total_containers < 1 or total_containers > 10:
            print("ERROR: container count out of range!")
            return None

    # Get the filled JSON for each container, 1 indexed
    reports = []
    for container_id_int in range(1, total_containers + 1):
        report_json["value"] = report_type.value
        glass_amount_json["value"] = total_containers
        container_id_json["value"] = container_id_int

        # Generate report for glass container_id_int
        generated_report = generate_container(
            input_text,
            report_type,
            str(container_id_int),
            copy.deepcopy(metadata_json),
            token_options,
        )

        reports.append(generated_report)

    return reports


def generate_container(
    input_text: str,
    report_type: ReportType,
    container_id: str,
    metadata_json: list[dict],
    token_options: TokenOptions = None,
) -> dict:
    """Function to generate a single JSON for container_id."""
    # Load the generated JSON template based on the report type
    template_json: list[dict] = load_json(
        f"data_model/out/generated-{report_type.value}.json"
    )

    generated_report = {
        "input_text": input_text,
        "target_json": [],
        "metadata_json": metadata_json,
    }

    # Process the template in batches to handle large templates
    # NOTE: If the len is always template_json then this can be simplified, as it seems to be within memory limits
    # For metadata/single queries it is important to set report_type to None
    batch_size = len(template_json)
    for i in range(0, len(template_json), batch_size):
        # Determine if caching should be used, only if the full template is used
        report_type = report_type if batch_size == len(template_json) else None
        template_batch = copy.deepcopy(template_json[i : i + batch_size])

        # Set the optional parameters for the token options
        if token_options is None:
            token_options = TokenOptions(report_type=report_type)
        else:
            token_options.report_type = report_type

        batch_filled = fill_json(
            TemplateGeneration(input_text, container_id, template_batch),
            token_options,
        )
        generated_report["target_json"].extend(batch_filled)
    return generated_report


@app.route("/models", methods=["GET"])
def get_models_endpoint():
    """Endpoint to get all available models from the configuration."""
    # Convert ModelSettings objects to string lists for JSON serialization
    serialized_models = {}
    for model_type, models in MODELS_DICT.items():
        serialized_models[model_type.value] = [str(model) for model in models]
    return jsonify(serialized_models)


@app.route("/load_model", methods=["POST"])
def load_model_endpoint():
    """Endpoint to load the specified model based on type."""
    model_type_str: str | None = request.json.get("model_type")
    model_index: int | None = request.json.get("model_index")
    is_trained: bool | None = request.json.get("is_trained")

    if model_index is None:
        return (jsonify({"error": "Model index is required"}), 400)
    if not model_type_str:
        return (
            jsonify(
                {"error": "Model type is required (decoder, encoder, encoder_decoder)"}
            ),
            400,
        )
    if model_type_str not in ModelType.get_enum_map():
        print("Invalid model type")
        return jsonify({"error": "Invalid model type"}), 400

    model_type = ModelType(model_type_str)
    if model_index < 0 or model_index >= len(MODELS_DICT[model_type]):
        return (
            jsonify({"error": "Model index out of range"}),
            404,
        )
    if is_trained is None:
        is_trained = False

    # Encoder models needs to be trained
    if model_type == ModelType.ENCODER:
        is_trained = True
    # Prevent trying to load local models that are not trained (e.g. Gemma and Qwen)
    elif model_type == ModelType.DECODER and model_index >= 2:
        is_trained = False

    return jsonify({"message": load_model(model_type, model_index, is_trained)}), 200


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
    if report_type_str:
        if report_type_str in ReportType.get_enum_map():
            report_type = ReportType(report_type_str)
        elif report_type_str == "diagnose":
            report_type = ReportType.MIKROSKOPISK
    total_containers: int | None = request.json.get("total_containers")

    # Set custom token options
    token_options = TokenOptions()
    include_enums = request.json.get("include_enums")
    if include_enums is not None:
        if not isinstance(include_enums, bool):
            return jsonify({"error": "include_enums must be a boolean"}), 400
        token_options.include_enums = include_enums

    generate_strings = request.json.get("generate_strings")
    if generate_strings is not None:
        if not isinstance(generate_strings, bool):
            return jsonify({"error": "generate_strings must be a boolean"}), 400
        token_options.generate_strings = generate_strings

    reports = generate(input_text, report_type, total_containers, token_options)
    if reports is None:
        return jsonify({"error": "Failed to generate structured data"}), 500

    return jsonify(reports)


@app.route("/correct/<string:report_id>", methods=["POST"])
def correct_endpoint(report_id: str):
    """Endpoint to save the corrected JSON by the user."""
    reports: list[dict] = request.json
    if not reports:
        return jsonify({"error": "No reports provided"}), 400

    # Generate a new report ID if not provided
    random_id = None
    if not report_id or report_id == "null":
        random_id = str(uuid.uuid4())
        report_id = random_id

    # Check if the report is already labeled
    labeled_ids_json: dict = load_json(LABELED_IDS_PATH)

    # Save every JSON in the list as a separate file
    last_type = None
    is_diagnose = False
    is_type_changed = False
    for report in reports:
        # Verify metadata_json structure
        if not report["metadata_json"] or len(report["metadata_json"]) < 3:
            return jsonify({"error": "Invalid report format"}), 400
        for item in report["metadata_json"]:
            if "value" not in item:
                return jsonify({"error": "Invalid report format"}), 400

        # Extract metadata values
        report_type_str: str = report["metadata_json"][0]["value"]
        total_glass_amount: int = report["metadata_json"][1]["value"]
        container_id: int = report["metadata_json"][2]["value"]

        if is_type_changed:
            is_type_changed = False
            # If the count the glass count is finished, but the report type is the same, its a diagnose text
            if last_type == report_type_str:
                is_diagnose = True
            last_type = report_type_str

        # Flag that the report type has changed
        if container_id == total_glass_amount:
            is_type_changed = True

        # Verify if the report type is valid
        if report_type_str not in ReportType.get_enum_map():
            return jsonify({"error": "Invalid report type"}), 400

        save_path = (
            f"{CORRECTED_OUT_DIR}{report_type_str}_{report_id}_{container_id}.json"
        )
        diag_path = (
            f"{CORRECTED_OUT_DIR}{report_type_str}_{report_id}_diag_{container_id}.json"
        )

        # Detect labeling diagnose text on a refreshed session.
        if (
            report_type_str == ReportType.MIKROSKOPISK.value
            and os.path.exists(save_path)
            and not os.path.exists(diag_path)
            and len(reports) == total_glass_amount
        ):
            is_diagnose = True

        if is_diagnose:
            save_path = diag_path
            labeled_type = DatasetField.DIAGNOSE.value
        else:
            # Find the matching report type in the DatasetField enum
            labeled_type = next(
                (
                    member.value
                    for member in DatasetField
                    if report_type_str.upper() in member.name
                ),
                None,
            )
        save_json(report, save_path)

        # Update which report type was labeled in the dataset
        if not random_id:
            labeled_ids_json.setdefault(report_id, {})[labeled_type] = True

    # Update the labeled IDs JSON
    save_json(labeled_ids_json, LABELED_IDS_PATH)

    return jsonify({"message": "Correctly labeled JSON saved!"})


@app.route("/unlabeled/<string:text_type_str>", methods=["GET"])
def unlabeled_endpoint(text_type_str: str):
    """
    Endpoint to get an unlabeled case of specified type.

    TODO:
        Keep in mind that multiple people could get the same case at the same time!
        For a distributed system, each user should get a seperate case that is not finished labeled.
        For a single user scenario like this PoC, this is not a problem.
        As its important to be able to continue where you left off on session refresh.
    """
    unlabeled_data = load_json(UNLABELED_BATCH_PATH)
    labeled_ids = load_json(LABELED_IDS_PATH)

    # Determine report type - handle "diagnose" as special case of MIKROSKOPISK
    is_diagnose = text_type_str == "diagnose"
    report_type = None
    if is_diagnose:
        report_type = ReportType.MIKROSKOPISK
    elif text_type_str in ReportType.get_enum_map():
        report_type = ReportType(text_type_str)

    # Define field mappings between report types and dataset fields
    field_map = {
        ReportType.KLINISK: DatasetField.KLINISK.value,
        ReportType.MAKROSKOPISK: DatasetField.MAKROSKOPISK.value,
        ReportType.MIKROSKOPISK: (
            DatasetField.MIKROSKOPISK.value
            if not is_diagnose
            else DatasetField.DIAGNOSE.value
        ),
    }

    # Find suitable unlabeled entry
    for entry in unlabeled_data:
        entry_id = entry["id"]
        labeled_info = labeled_ids.get(entry_id, {})

        # Skip fully labeled entries
        if all(labeled_info.get(field, False) for field in DatasetField.get_enum_map()):
            continue

        # For entries without a report type specified
        if not report_type:
            # Default to KLINISK for new entries
            if not labeled_info:
                report_type = ReportType.KLINISK
            else:
                # Find first unlabeled report type
                for rt in ReportType:
                    field = field_map.get(rt)
                    if not labeled_info.get(field, False) and entry.get(field):
                        report_type = rt
                        break

                # Check for unlabeled diagnose as special case
                if (
                    not labeled_info.get(DatasetField.DIAGNOSE.value, False)
                    and report_type is None
                ):
                    report_type = ReportType.MIKROSKOPISK
                    field_map[report_type] = DatasetField.DIAGNOSE.value
                    is_diagnose = True

        # Process the selected report type
        if report_type:
            field = field_map.get(report_type)

            # Skip if already labeled
            if labeled_info.get(field, False):
                continue

            # Get and return the report text if available
            report_text = entry.get(field)
            if report_text:
                return jsonify(
                    {
                        "id": entry_id,
                        "is_diagnose": is_diagnose,  # Value needed for the frontend auto setting of report type
                        "report_type": report_type.value,
                        "text": report_text.strip().replace("\r", "\n"),
                    }
                )

    # No unlabeled cases found
    return jsonify({"error": "No unlabeled cases found!"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
