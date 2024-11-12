import json
from model_loader import ModelLoader

models_dict = {
    "encoder-decoder": "google/flan-t5-base",
    "decoder": "mistralai/Mistral-7B-Instruct-v0.3",
    "encoder": "google-bert/bert-base-uncased",
}

if __name__ == "__main__":
    model_type = "encoder-decoder"

    text = "The car is a 2021 model and comes in a red color."
    json_template = [
        {"id": 1, "field": "model_year", "value": "FILL_VALUE", "required": True},
        {"id": 2, "field": "color", "value": "FILL_VALUE", "required": True},
    ]

    model_loader = ModelLoader(models_dict[model_type], model_type)
    filled_json = model_loader.generate_filled_json(text, json_template)

    print("Filled JSON:\n", json.dumps(filled_json, indent=2))
