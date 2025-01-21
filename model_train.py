from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from model_loader import ModelLoader
from config import MODELS_DICT
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


def train_model(loader: ModelLoader, training_data: Dataset, output_dir: str):
    """
    Train the model using the provided dataset.
    :param loader: ModelLoader object with the loaded model and tokenizer.
    :param training_data: Tokenized training dataset.
    :param output_dir: Directory to save the fine-tuned model.
    """
    # Define training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",  # Set to "epoch"
        learning_rate=2e-4,
        num_train_epochs=50,  # Make selectable along with other training params
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,  # Enable mixed precision
    )

    print(training_data)
    if loader.model_type == "encoder-decoder":
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=loader.tokenizer,
            return_tensors="pt",
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=loader.tokenizer,
            mlm=False,
            return_tensors="pt",
        )

    trainer = Trainer(
        model=loader.model,
        args=training_args,
        train_dataset=training_data,
        processing_class=loader.tokenizer,
        # eval_dataset= # Add evaluation dataset and set eval_strategy to "epoch"
        data_collator=data_collator,
    )

    # Train the model.
    trainer.train()

    # Save the fine-tuned model.
    loader.model.save_pretrained(output_dir)
    loader.tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    MODEL_TYPE = "encoder-decoder"

    # Load model and tokenizer.
    model_loader = ModelLoader(MODELS_DICT[MODEL_TYPE], MODEL_TYPE)

    if MODEL_TYPE == "decoder":
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,  # Rank of the LoRA layers
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj"],  # Which layers to apply LoRA
            lora_dropout=0.1,  # Dropout for LoRA
            bias="none",  # No bias in LoRA layers
        )

        peft_model = get_peft_model(model_loader.model, lora_config)
        peft_model.print_trainable_parameters()

        # Apply PEFT to the decoder model
        model_loader.model = peft_model

    # Load the test dataset.
    dataset = dataset_loader.create_dataset(
        "data/labeled_data/test/",
        model_loader.model_type,
        model_loader.tokenizer.mask_token,
    )

    print("\n\n")
    print(dataset["output"][0])

    output_size = len(dataset["output"])
    print(f"Number of examples: {output_size}")

    # Tokenize the dataset.
    tokenized_dataset = tokenize_dataset(model_loader.tokenizer, dataset)

    print(len(tokenized_dataset["input_ids"][0]))
    print(len(tokenized_dataset["labels"][0]))

    # labels = tokenized_dataset["labels"]
    # labels = labels[labels == model_loader.tokenizer.pad_token_id] = -100

    # training_data = tokenized_dataset.train_test_split(test_size=0.1)

    # Train/Fine-tune and save the model.
    train_model(model_loader, tokenized_dataset, f"trained/{MODEL_TYPE}")
