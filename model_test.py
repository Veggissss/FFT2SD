import json
from model_loader import ModelLoader
from config import MODELS_DICT

if __name__ == "__main__":
    # Define the model type and whether it is trained or not.
    model_type = "decoder"
    is_trained = True

    # Load the json template for clinical data report
    with open("data_model/out/generated-klinisk.json", "r") as f:
        template_json = json.load(f)

    # Load the input text and the correct output
    with open("data/test/test-klinisk.json", "r") as f:
        test_json = json.load(f)

    # Add 'trained-' prefix to the model name if it is trained.
    model_key = f"trained-{model_type}" if is_trained else model_type
    model_loader = ModelLoader(MODELS_DICT[model_key], model_type)

    for template_entry in template_json:
        template_str = json.dumps(template_entry)
        print(template_str)
        template_str = template_str.replace(
            '"value": null', f'"value": {model_loader.tokenizer.mask_token}'
        )

        print(template_str)

        # Generate filled JSON using the model
        filled_json = model_loader.generate_filled_json(
            test_json["input_text"], template_str
        )

        print("Filled JSON:\n", json.dumps(filled_json, indent=2))
