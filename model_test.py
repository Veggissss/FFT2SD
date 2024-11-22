import json
from model_loader import ModelLoader
from config import MODELS_DICT

if __name__ == "__main__":
    model_type = "decoder"
    is_trained = False

    text = "The car is a 1995 model and comes in a blue color."
    template_json = json.dumps(
        [
            {"id": 1, "field": "year", "value": None, "required": True},
            {"id": 2, "field": "color", "value": None, "required": True},
        ]
    )

    # Add 'trained-' prefix to the model name if it is trained.
    model_key = f"trained-{model_type}" if is_trained else model_type

    model_loader = ModelLoader(MODELS_DICT[model_key], model_type)

    filled_json = model_loader.generate_filled_json(text, template_json)

    print("Filled JSON:\n", json.dumps(filled_json, indent=2))
