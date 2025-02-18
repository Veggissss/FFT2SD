from typing import Literal
import json
from flask import Flask, request, jsonify

from config import MODELS_DICT, CONTAINER_NUMBER_MASK
from model_loader import ModelLoader

app = Flask(__name__)

# Global variable to store the loaded model
IS_TRAINED = True
model_loader = None


def load_json(file_path: str) -> dict:
    """Function to load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_type: Literal["decoder", "encoder", "encoder-decoder"]) -> str:
    """Function to load the specified LLM model based on type."""
    model_key = f"trained-{model_type}" if IS_TRAINED else model_type

    # Update the global model_loader variable
    global model_loader
    model_loader = ModelLoader(MODELS_DICT[model_key], model_type)

    return f"Loaded model: {model_type}"


def generate(
    input_text: str, report_type: Literal["klinisk", "makroskopisk", "mikroskopisk"]
) -> dict:
    """Function to generate text using the loaded model."""
    mask_token = (
        model_loader.tokenizer.mask_token
        if model_loader.tokenizer.mask_token is not None
        else "null"
    )

    # Mask out the total amount of containers present in the input text
    metadata_json = load_json("data_model/out/generated-metadata.json")
    metadata_json[1]["value"] = f"{mask_token}"

    # Find out how many containers there are in the input text
    filled_json = model_loader.generate_filled_json(
        input_text,
        CONTAINER_NUMBER_MASK,
        json.dumps(metadata_json[1], ensure_ascii=False),
    )

    if filled_json.get("value") is not None:
        try:
            total_containers = int(filled_json["value"])
        except (ValueError, TypeError) as e:
            print(f"ERROR: Could not parse the container count! {e}")
            total_containers = 1  # DEBUG VALUE
    else:
        print("ERROR: Could not find the container count!")
        return None

    print(f"Container count: {total_containers}")
    if total_containers < 1 or total_containers > 10:
        print("ERROR: Invalid container count!")
        return None

    # Load the generated JSON template based on the report type
    template_json = load_json(f"data_model/out/generated-{report_type}.json")
    final_json = {"input": input_text, "output": []}

    # Get the filled JSON for each container, 1 indexed
    for container_number in range(1, total_containers + 1):
        template_filled_jsons = {f"glass {container_number}": []}
        for template_entry in template_json:
            # Mask out the value in the template
            template_entry["value"] = str(mask_token)
            template_str = json.dumps(template_entry, indent=2, ensure_ascii=False)

            # Generate filled JSON using the model
            filled_json = model_loader.generate_filled_json(
                input_text, container_number, template_str
            )
            template_filled_jsons[f"glass {container_number}"].append(filled_json)
        final_json["output"].append(template_filled_jsons)

    return final_json


@app.route("/load_model", methods=["POST"])
def load_model_endpoint():
    """Endpoint to load the specified model based on type."""
    model_type = request.json.get("model_type")
    if not model_type:
        return (
            jsonify(
                {"error": "Model type is required (decoder, encoder, encoder-decoder)"}
            ),
            400,
        )
    if model_type not in ["decoder", "encoder", "encoder-decoder"]:
        return jsonify({"error": "Invalid model type"}), 400

    return jsonify({"message": load_model(model_type)})


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    """Endpoint to generate structured data as json based on text input."""
    if model_loader is None:
        return jsonify({"error": "Model is not loaded!"}), 400

    input_text = request.json.get("input_text")
    if not input_text:
        print("Input text is required")
        return jsonify({"error": "Input text is required"}), 400

    report_type = request.json.get("report_type")
    if not report_type:
        return (jsonify({"error": "Report type is required"}),)
    if report_type not in ["klinisk", "makroskopisk", "mikroskopisk"]:
        return jsonify({"error": "Invalid report type"}), 400

    print(f"Input text: {input_text}")

    final_json = generate(input_text, report_type)
    if final_json is None:
        return jsonify({"error": "Failed to generate structured data"}), 500

    return jsonify(final_json)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
