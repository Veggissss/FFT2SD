from model_loader import ModelLoader
from config import MODELS_DICT
import json

if __name__ == "__main__":
    # Define the model type and whether it is trained or not.
    MODEL_TYPE = "encoder"
    IS_TRAINED = True

    # Load the test clinical data report data
    with open(
        "data/test_data/container_0_case_1_diagn.json", "r", encoding="utf-8"
    ) as f:
        test_json = json.load(f)

    with open("data_model/out/generated-metadata.json", "r", encoding="utf-8") as f:
        metadata_json = json.load(f)

    # Add 'trained-' prefix to the model name if it is trained.
    model_key = f"trained-{MODEL_TYPE}" if IS_TRAINED else MODEL_TYPE
    model_loader = ModelLoader(MODELS_DICT[model_key], MODEL_TYPE)

    mask_token = (
        model_loader.tokenizer.mask_token
        if model_loader.tokenizer.mask_token is not None
        else "null"
    )

    # Mask out the total amount of containers present in the input text
    metadata_json[1]["value"] = f"{mask_token}"

    # First find out how many containers/Beholder-IDs there are in the input text
    filled_json = model_loader.generate_filled_json(
        test_json["input_text"],
        "?",
        json.dumps(metadata_json[1], ensure_ascii=False),
    )

    if filled_json.get("value") is not None:
        TOTAL_CONTAINERS = int(filled_json["value"])
        print(f"Container count: {TOTAL_CONTAINERS}")
    else:
        print("ERROR: Could not find the container count!")
        TOTAL_CONTAINERS = 1  # DEBUG VALUE

    # Find the report type for the input text
    metadata_json[0]["value"] = f"{mask_token}"

    filled_json = model_loader.generate_filled_json(
        test_json["input_text"],
        "1",
        json.dumps(metadata_json[0], ensure_ascii=False),
    )

    if filled_json.get("value") is not None:
        REPORT_TYPE = str(filled_json["value"]).strip()
        print(f"Container count: {TOTAL_CONTAINERS}")
    else:
        REPORT_TYPE = "klinisk"  # DEBUG VALUE
        print("ERROR: Could not find the container count!")

    with open(
        f"data_model/out/generated-{REPORT_TYPE}.json", "r", encoding="utf-8"
    ) as f:
        template_json = json.load(f)

    for container_number in range(TOTAL_CONTAINERS):
        for template_entry in template_json:
            template_entry["value"] = str(mask_token)
            template_str = json.dumps(template_entry, indent=2, ensure_ascii=False)

            # Generate filled JSON using the model
            filled_json = model_loader.generate_filled_json(
                test_json["input_text"], container_number + 1, template_str
            )

            print(
                "Filled JSON:\n", json.dumps(filled_json, indent=2, ensure_ascii=False)
            )
