# Definition of different Hugging Face models that are to be trained and TODO: local trained models paths.
MODELS_DICT: dict[str, str] = {
    "trained-encoder-decoder": "fine_tuned_model",
    "encoder-decoder": "google/flan-t5-base",
    "decoder": "mistralai/Mistral-7B-Instruct-v0.3",
    "encoder": "google-bert/bert-base-uncased",
}
