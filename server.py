from typing import Literal
from flask import Flask, request, jsonify

from config import MODELS_DICT, CONTAINER_NUMBER_MASK
from model_loader import ModelLoader
from file_loader import load_json, json_to_str

app = Flask(__name__)

# Global variable to store the loaded model
model_loader = None
IS_TRAINED = True


def load_model(model_type: Literal["decoder", "encoder", "encoder-decoder"]) -> str:
    """Function to load the specified LLM model based on type."""
    model_key = f"trained-{model_type}" if IS_TRAINED else model_type
    model_name = MODELS_DICT[model_key]

    # Update the global model_loader variable
    global model_loader
    model_loader = ModelLoader(model_name, model_type)

    return f"Loaded model: {model_name} | {model_type}"


def generate(input_text: str) -> dict | None:
    """Function to generate text using the loaded model."""
    mask_token: str = (
        model_loader.tokenizer.mask_token
        if model_loader.tokenizer.mask_token is not None
        else "null"
    )

    # Mask out the total amount of containers present in the input text
    metadata_json = load_json("data_model/out/generated-metadata.json")
    metadata_json[0]["value"] = mask_token
    metadata_json[1]["value"] = mask_token

    # Find out the report type based on the input text
    filled_json = model_loader.generate_filled_json(
        input_text, CONTAINER_NUMBER_MASK, json_to_str(metadata_json[0], indent=2)
    )
    report_type = filled_json.get("value", "null").strip()
    print(f"Report type: {report_type}")
    if report_type not in ["klinisk", "makroskopisk", "mikroskopisk"]:
        print("ERROR: Invalid report type!")
        return None

    # Find out how many containers there are in the input text
    filled_json = model_loader.generate_filled_json(
        input_text, CONTAINER_NUMBER_MASK, json_to_str(metadata_json[1], indent=2)
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
    final_json = {"input": input_text, "output": [], "report_type": report_type}

    # Get the filled JSON for each container, 1 indexed
    for container_number in range(1, total_containers + 1):
        template_filled_jsons = {f"glass {container_number}": []}
        for template_entry in template_json:
            # Mask out the value in the template
            template_entry["value"] = str(mask_token)
            template_str = json_to_str(template_entry, indent=2)

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

    final_json = generate(input_text)
    if final_json is None:
        return jsonify({"error": "Failed to generate structured data"}), 500

    return jsonify(final_json)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
