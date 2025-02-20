from peft import LoraConfig, get_peft_model
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
from enums import ModelType
import dataset_loader


def tokenize_dataset(tokenizer: AutoTokenizer, text_data: Dataset) -> Dataset:
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
            return_tensors="np",  # NumPy is faster here: https://huggingface.co/docs/datasets/nlp_process#map
        ),
        batched=True,
    )


def train_model(loader: ModelLoader, training_data: Dataset, output_dir: str) -> None:
    """
    Train the model using the provided dataset.
    :param loader: ModelLoader object with the loaded model and tokenizer.
    :param training_data: Tokenized training dataset.
    :param output_dir: Directory to save the fine-tuned model.
    """
    # Remove untokenized input and output columns.
    training_data = training_data.remove_columns(["input", "output"])
    training_data = training_data.train_test_split(test_size=0.1)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        num_train_epochs=5,  # TODO: Make selectable along with other training params
        # learning_rate=2e-4,
        # weight_decay=0.01,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,  # Mixed precision
    )

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

    trainer = Trainer(
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


def train(model_type: ModelType) -> None:
    """
    Train the model using the provided dataset.
    :param model_type: Model type to train.
    """
    # Load the untrained model and tokenizer.
    model_loader = ModelLoader(model_type, is_trained=False)

    # Quantized models can't be trained directly.
    if model_loader.model_type == ModelType.DECODER:
        # Configure LoRA
        lora_config = LoraConfig(
            r=128,  # Rank of the LoRA layers
            lora_alpha=256,  # Scaling factor
            lora_dropout=0.1,  # Dropout for LoRA
            bias="none",  # No bias in LoRA layers
            task_type="CAUSAL_LM",
        )

        # Apply PEFT to the decoder model
        peft_model = get_peft_model(model_loader.model, lora_config)
        peft_model.print_trainable_parameters()

        # Use the PEFT model for training
        model_loader.model = peft_model

    # Load the test dataset.
    dataset, enums = dataset_loader.create_dataset(
        "data/test_data/", model_loader.model_type
    )
    example_count = len(dataset["input"])
    print(f"Number of examples: {example_count}")

    # Register enum strings present in dataset as new tokens.
    print(f"Number of tokens in the tokenizer before: {len(model_loader.tokenizer)}")

    # Make sure all the used datatypes are present in the tokenizer
    enums.extend(["true", "false"])
    print(f"New tokens: \n{enums}")
    new_tokens = [
        AddedToken(enum, single_word=True, rstrip=True, lstrip=True) for enum in enums
    ]

    # Add tokens to the tokenizer.
    model_loader.tokenizer.add_tokens(new_tokens)
    model_loader.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Resize the model's token embeddings to fit the new tokens.
    model_loader.model.resize_token_embeddings(len(model_loader.tokenizer))
    print(f"Number of tokens in the tokenizer: {len(model_loader.tokenizer)}")

    # Fix for: "CUDA Assertion `t >= 0 && t < n_classes` failed" for the ltg encoder and encoder-decoder models
    # The Classifier does not get resized when calling model.resize_token_embeddings() so needs to be manually re-initialized
    if model_loader.model_type == ModelType.ENCODER:
        model_loader.model.classifier.__init__(
            model_loader.model.config,
            model_loader.model.embedding.word_embedding.weight,
        )
    elif model_loader.model_type == ModelType.ENCODER_DECODER:
        model_loader.model.classifier.__init__(model_loader.model.config)

    # Tokenize the dataset.
    tokenized_dataset = tokenize_dataset(model_loader.tokenizer, dataset)
    print(tokenized_dataset)

    # Train/Fine-tune and save the model.
    train_model(
        model_loader, tokenized_dataset, f"trained/{model_loader.model_type.value}"
    )


if __name__ == "__main__":
    TRAIN_ALL_TYPES = True
    if TRAIN_ALL_TYPES:
        for model_type in ModelType:
            train(model_type)
    else:
        train(ModelType.ENCODER)
