from model_loader import ModelLoader
from config import MODELS_DICT
import json

if __name__ == "__main__":
    # Define the model type and whether it is trained or not.
    MODEL_TYPE = "decoder"
    IS_TRAINED = True

    # Load the test clinical data report data
    with open(
        "data/test_data/container_0_case_1_diagn.json", "r", encoding="utf-8"
    ) as f:
        test_json = json.load(f)

    # Add 'trained-' prefix to the model name if it is trained.
    model_key = f"trained-{MODEL_TYPE}" if IS_TRAINED else MODEL_TYPE
    model_loader = ModelLoader(MODELS_DICT[model_key], MODEL_TYPE)

    mask_token = (
        model_loader.tokenizer.mask_token
        if model_loader.tokenizer.mask_token is not None
        else "[UTFYLL]"
    )
    glass_count_entry = {
        "id": 0,
        "field": "Antall glass",
        "type": "int",
        "value": f"{mask_token}",
    }

    # First find out how many containers/Beholder-IDs there are in the input text
    filled_json = model_loader.generate_filled_json(
        test_json["input_text"],
        "?",
        json.dumps(glass_count_entry, ensure_ascii=False),
    )
    # print(filled_json)

    if filled_json.get("value") is not None:
        TOTAL_CONTAINERS = int(filled_json["value"])
        print(f"Container count: {TOTAL_CONTAINERS}")
    else:
        print("Could not find the container count.")
        TOTAL_CONTAINERS = 1  # DEBUG VALUE

    for container_number in range(TOTAL_CONTAINERS):
        for template_entry in test_json["template_json"]:
            template_entry["value"] = str(mask_token)
            template_str = json.dumps(template_entry, indent=2, ensure_ascii=False)
            # print(template_str)

            # Generate filled JSON using the model
            filled_json = model_loader.generate_filled_json(
                test_json["input_text"], container_number + 1, template_str
            )

            print(
                "Filled JSON:\n", json.dumps(filled_json, indent=2, ensure_ascii=False)
            )
