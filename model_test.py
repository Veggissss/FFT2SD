from model_loader import ModelLoader
from utils.enums import ModelType
from utils.file_loader import load_text, json_to_str
import server

if __name__ == "__main__":
    MODEL_TYPE = ModelType.ENCODER
    IS_TRAINED = True
    server.model_loader = ModelLoader(MODEL_TYPE, IS_TRAINED)

    input_text = load_text("data/example_batch/case_5_klinisk_oppl.txt")
    print(input_text)

    out = server.generate(input_text)
    print(json_to_str(out))
