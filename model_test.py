from model_loader import ModelLoader
from utils.enums import ModelType
from utils.file_loader import load_json, json_to_str
import server

if __name__ == "__main__":
    MODEL_TYPE = ModelType.ENCODER_DECODER
    IS_TRAINED = True
    server.model_loader = ModelLoader(MODEL_TYPE, IS_TRAINED)

    test_data = load_json("data/test_data/case_1_diagn_1.json")
    input_text = test_data["input_text"]
    print(input_text)

    out = server.generate(input_text)
    print(json_to_str(out))
