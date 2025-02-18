from model_loader import ModelLoader
from config import MODELS_DICT
from file_loader import load_json, json_to_str


def initialize_model(model_type, is_trained):
    """Initialize the model loader based on the model type and training status."""
    model_key = f"trained-{model_type}" if is_trained else model_type
    return ModelLoader(MODELS_DICT[model_key], model_type)


def get_mask_token(model_loader: ModelLoader):
    """Retrieve the mask token from the tokenizer, defaulting to 'null'."""
    return model_loader.tokenizer.mask_token or "null"


def get_filled_json(
    model_loader: ModelLoader, input_text: str, container_str: str, template
):
    """Generate filled JSON using the model."""
    return model_loader.generate_filled_json(
        input_text, container_str, json_to_str(template)
    )


if __name__ == "__main__":
    MODEL_TYPE = "encoder"
    IS_TRAINED = True

    test_data = load_json("data/test_data/container_0_case_1_diagn.json")
    metadata = load_json("data_model/out/generated-metadata.json")
    model_loader = initialize_model(MODEL_TYPE, IS_TRAINED)

    # Encoder models need a mask token to generate output
    mask_token = get_mask_token(model_loader)

    # Determine the report type
    metadata[0]["value"] = mask_token
    filled_json = model_loader.generate_filled_json(
        test_data["input_text"], "1", json_to_str(metadata[0])
    )
    report_type = filled_json.get("value", "klinisk").strip()
    print(f"Report type: {report_type}")

    # Load the corresponding template
    template_data = load_json(f"data_model/out/generated-{report_type}.json")

    # Determine the total container count
    metadata[1]["value"] = mask_token
    filled_json = model_loader.generate_filled_json(
        test_data["input_text"], "?", json_to_str(metadata[1])
    )
    total_containers = int(filled_json.get("value", 1))
    print(f"Container count: {total_containers}")

    # Process each container entry
    for container_id in range(1, total_containers + 1):
        for entry in template_data:
            entry["value"] = mask_token
            filled_json = model_loader.generate_filled_json(
                test_data["input_text"], container_id, json_to_str(entry)
            )
            print("Filled JSON:\n", json_to_str(filled_json, indent=2))
