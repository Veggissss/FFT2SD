from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AddedToken,
)
from datasets import Dataset
from model_loader import ModelLoader
from config import JSON_START_MARKER, MODELS_DICT
from utils.enums import ModelType
import dataset_loader


def tokenize_dataset(
    tokenizer: AutoTokenizer, text_data: Dataset, max_length: int
) -> Dataset:
    """
    Tokenize the dataset using the provided model tokenizer.
    :param tokenizer: Model tokenizer.
    :param text_data: Dataset to tokenize.
    :return: Tokenized dataset.
    """
    return text_data.map(
        lambda data: tokenizer(
            data["input"],
            text_target=data["output"],
            padding=True,
            max_length=max_length,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="np",  # NumPy is faster here: https://huggingface.co/docs/datasets/nlp_process#map
        ),
        batched=True,
    )


def train_model(
    loader: ModelLoader,
    training_data: Dataset,
    output_dir: str,
    training_args: TrainingArguments,
) -> None:
    """
    Train the model using the provided dataset.
    :param loader: ModelLoader object with the loaded model and tokenizer.
    :param training_data: Tokenized training dataset.
    :param output_dir: Directory to save the fine-tuned model.
    """
    # Remove untokenized input and output columns.
    training_data = training_data.remove_columns(["input", "output"])
    training_data = training_data.train_test_split(test_size=0.01)

    match loader.model_type:
        case ModelType.ENCODER_DECODER:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=loader.tokenizer,
                return_tensors="pt",
            )
        case ModelType.ENCODER:
            training_data = training_data.remove_columns(["labels"])
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=loader.tokenizer,
                mlm=True,
                mlm_probability=0.15,
                return_tensors="pt",
            )
        case ModelType.DECODER:
            training_data = training_data.remove_columns(["labels"])
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=loader.tokenizer,
                mlm=False,
                return_tensors="pt",
            )

    # null_token_id = loader.tokenizer.convert_tokens_to_ids("null")
    # weights = torch.ones(len(loader.tokenizer), dtype=torch.float32)
    # weights[null_token_id] = 0.2

    trainer = Trainer(
        # weights=weights,
        model=loader.model,
        args=training_args,
        train_dataset=training_data["train"],
        eval_dataset=training_data["test"],
        processing_class=loader.tokenizer,
        data_collator=data_collator,
    )

    # Train the model.
    trainer.train()

    # Save the fine-tuned model and tokenizer.
    loader.model.save_pretrained(output_dir)
    loader.tokenizer.save_pretrained(output_dir)


def apply_peft(model_loader: ModelLoader) -> PeftModel:
    """
    Apply the PEFT model to the decoder model.
    """
    # Configure LoRA
    lora_config = LoraConfig(
        r=128,  # Rank of the LoRA layers
        lora_alpha=256,  # Scaling factor
        lora_dropout=0.1,  # Dropout for LoRA
        bias="none",  # No bias in LoRA layers
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model_loader.model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def add_tokens_to_tokenizer(model_loader: ModelLoader, enums: list[str]) -> None:
    """
    Add the used datatypes to the tokenizer and resize the model's token embeddings.
    """
    # Make sure all the used datatypes are present in the tokenizer
    enums.extend(["true", "false", JSON_START_MARKER, "id", "field", "type", "value"])
    new_tokens = [
        AddedToken(enum, single_word=True, rstrip=True, lstrip=True) for enum in enums
    ]

    # Add tokens to the tokenizer.
    model_loader.tokenizer.add_tokens(new_tokens)
    model_loader.tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )

    # Resize the model's token embeddings to fit the new tokens.
    model_loader.model.resize_token_embeddings(len(model_loader.tokenizer))

    # Fix for: "CUDA Assertion `t >= 0 && t < n_classes` failed" for the ltg encoder and encoder-decoder models
    # The Classifier does not get resized when calling model.resize_token_embeddings() so needs to be manually re-initialized
    if model_loader.model_type == ModelType.ENCODER:
        model_loader.model.classifier.__init__(
            model_loader.model.config,
            model_loader.model.embedding.word_embedding.weight,
        )
    elif model_loader.model_type == ModelType.ENCODER_DECODER:
        model_loader.model.classifier.__init__(model_loader.model.config)


def reinitialize_weights(module):
    """TODO: Simulate untrained model by reinitializing the weights of the model layers."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
        print(f"Reinitialized weights for {module.__class__.__name__}")


def train(model_type: ModelType, model_index: int) -> None:
    """
    Train the model using the provided dataset.
    :param model_type: Model type to train.
    """
    model_loader = ModelLoader(model_type, model_index, is_trained=False)
    dataset_dir = "data/corrected/"
    output_dir = f"trained/{model_loader.model_settings.__str__()}"
    batch_size = 3

    print(f"Saving trained model to: {output_dir}")

    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        learning_rate=4e-4,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,  # Mixed precision
        warmup_ratio=0.1,  # 10% warmup
        lr_scheduler_type="cosine",
    )

    # Load the dataset.
    dataset, enums = dataset_loader.create_dataset(dataset_dir, model_loader.model_type)
    print(dataset["input"][:2])
    dataset.batch(batch_size=batch_size)

    # Quantized models can't be trained directly.
    if model_loader.model_type == ModelType.DECODER:
        model_loader.model = apply_peft(model_loader)

    # Add new tokens if the model is not trained.
    if not model_loader.is_trained:
        add_tokens_to_tokenizer(model_loader, enums)
        print(f"Number of tokens in the tokenizer: {len(model_loader.tokenizer)}")

    # Tokenize the dataset.
    max_length = model_loader.model.config.max_position_embeddings
    tokenized_dataset = tokenize_dataset(model_loader.tokenizer, dataset, max_length)

    # Find the longest tensor in the tokenized_dataset["inputs"] column
    tensor_length = len(tokenized_dataset["input_ids"][0])
    print(f"Longest tensor length in 'input_ids' column: {tensor_length}")
    if tensor_length == max_length:
        print("Truncated the input tensors to the maximum length.")

    # Train/Fine-tune and save the model.
    train_model(model_loader, tokenized_dataset, output_dir, training_args)


if __name__ == "__main__":
    # Train all model types and sizes
    for m_type in ModelType:
        for i in range(len(MODELS_DICT[m_type])):
            train(m_type, i)
            # train(m_type, 1)
            # train(m_type, 2)
