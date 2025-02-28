from model_loader import ModelLoader
from utils.enums import ModelType
from utils.file_loader import load_text, json_to_str
import server

if __name__ == "__main__":
    MODEL_TYPE = ModelType.ENCODER_DECODER
    IS_TRAINED = True
    server.model_loader = ModelLoader(MODEL_TYPE, IS_TRAINED)

    # Simple manual test
    CASE_NUMBER = 1
    TEXT_TYPES = ["klinisk_oppl", "makro", "diagn"]
    TEXT_INDEX = 0

    input_text = load_text(
        f"data/example_batch/case_{CASE_NUMBER}_{TEXT_TYPES[TEXT_INDEX]}.txt"
    )
    print(input_text)

    out = server.generate(input_text)
    print(json_to_str(out))
