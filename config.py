# Definitions of different models
MODELS_DICT: dict[str, str] = {
    # Trained local models
    "trained-encoder-decoder": "trained/encoder-decoder",
    "trained-decoder": "trained/decoder",
    "trained-encoder": "trained/encoder",
    # Hugging Face models
    "encoder-decoder": "ltg/nort5-base",
    "decoder": "norallm/normistral-7b-warm-instruct",
    "encoder": "ltg/norbert3-base",
}

# Mark the end of the prompt for the model to start generating the output.
# Useful for decoder models which continue generating using the prompt.
END_OF_PROMPT_MARKER = "[MASK_8]"

# Prompt for filling in the null JSON values. Used in training and evaluation.
SYSTEM_PROMPT = (
    "Gitt teksten: '{input_text}'.\nFyll ut den manglede verdien for feltet \"value\", behold JSON strukturen: \n{template_json} "
    + END_OF_PROMPT_MARKER
)
