from model_loader import ModelLoader
from config import MODELS_DICT
import json

if __name__ == "__main__":
    # Define the model type and whether it is trained or not.
    MODEL_TYPE = "encoder"
    IS_TRAINED = True

    # Load the test clinical data report data
    with open(
        "data/labeled_data/test/container_0_case_1_diagn.json", "r", encoding="utf-8"
    ) as f:
        test_json = json.load(f)

    # Add 'trained-' prefix to the model name if it is trained.
    model_key = f"trained-{MODEL_TYPE}" if IS_TRAINED else MODEL_TYPE
    model_loader = ModelLoader(MODELS_DICT[model_key], MODEL_TYPE)

    glass_count_entry = {
        "id": 0,
        "field": "Antall glass",
        "type": "int",
        "value": f"{model_loader.tokenizer.mask_token}",
    }

    # First find out how many containers/Beholder-IDs there are in the input text
    filled_json = model_loader.generate_filled_json(
        test_json["input_text"], "", glass_count_entry
    )
    print(filled_json)

    container_count = 0
    if filled_json.get("value") is not None:
        container_count = filled_json["value"]
        print(f"Container count: {container_count}")
    else:
        print("Could not find the container count.")

    for container_number in range(container_count):
        for template_entry in test_json["template_json"]:
            template_str = json.dumps(template_entry)
            print(template_str)
            template_str = template_str.replace(
                '"value": null', f'"value": {model_loader.tokenizer.mask_token}'
            )

            print(template_str)

            # Generate filled JSON using the model
            filled_json = model_loader.generate_filled_json(
                test_json["input_text"], container_number + 1, template_str
            )

            print("Filled JSON:\n", json.dumps(filled_json, indent=2))
