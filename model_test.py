import json
from model_loader import ModelLoader

if __name__ == "__main__":
    model_name = "t5-small"
    model_type = "encoder-decoder"

    text = "The car is a 2021 model and comes in a red color."
    json_template = [
        {"id": 1, "field": "model_year", "value": None, "required": True},
        {"id": 2, "field": "color", "value": None, "required": True},
    ]

    model_loader = ModelLoader(model_name, model_type)
    filled_json = model_loader.generate_filled_json(text, json_template)

    print("Filled JSON:", json.dumps(filled_json, indent=2))
