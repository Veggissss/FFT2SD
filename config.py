# Definitions of different models
MODELS_DICT: dict[str, str] = {
    # Trained local models
    "trained-encoder-decoder": "trained/encoder-decoder",
    "trained-decoder": "trained/decoder",
    "trained-encoder": "trained/encoder",
    # Hugging Face models
    "encoder-decoder": "ltg/nort5-base",
    "decoder": "norallm/normistral-7b-warm",
    "encoder": "google-bert/bert-base-uncased",
}

# Mark the end of the prompt for the model to start generating the output.
# Useful for decoder models which continue generating using the prompt.
END_OF_PROMPT_MARKER = "<END_OF_PROMPT>"

# Prompt for filling in the null JSON values. Used in training and evaluation.
SYSTEM_PROMPT = (
    "Input Text: '{input_text}'\nFill in the null JSON values while retaining its structure:\n{template_json}\n"
    + END_OF_PROMPT_MARKER
)
