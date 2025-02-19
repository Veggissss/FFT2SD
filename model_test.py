from model_loader import ModelLoader
from enums import ModelType
from file_loader import load_json
import server

if __name__ == "__main__":
    MODEL_TYPE = ModelType.ENCODER
    IS_TRAINED = True
    server.model_loader = ModelLoader(MODEL_TYPE, IS_TRAINED)

    test_data = load_json("data/test_data/container_0_case_1_diagn.json")
    input_text = test_data["input_text"]

    out = server.generate(input_text)
