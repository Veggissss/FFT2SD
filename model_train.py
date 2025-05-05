from dotenv import dotenv_values
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AddedToken,
    EarlyStoppingCallback,
)
from transformers.data.data_collator import DataCollatorMixin
from datasets import Dataset
from model_loader import ModelLoader
from config import JSON_START_MARKER, MODELS_DICT
from utils.enums import ModelType
from dataset_loader import DatasetLoader
import torch


class DataCollatorForMaskedValueTokens(DataCollatorMixin):
    """
    Data collator to make the labels the same length as the input_ids.

    NOTE: Regular DataCollatorWithPadding seems to not work with the labels.
    Could pad the labels to max_length, but it would be a waste of memory.
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, dataset: list[dict[str, torch.LongTensor]], return_tensors="pt"
    ) -> dict[str, torch.LongTensor]:
        # Convert from lazy rows to list
        input_ids = [sample["input_ids"] for sample in dataset]
        labels = [sample["labels"] for sample in dataset]
        batch = self.tokenizer.pad(
            {"input_ids": input_ids}, return_tensors=return_tensors
        )

        # Pad the labels to the same length as input_ids, use -100 to ignore the loss
        max_len = batch["input_ids"].size(1)
        batch["labels"] = torch.tensor(
            [label + ([-100] * (max_len - len(label))) for label in labels],
            dtype=torch.long,
        )
        return batch


def mask_value_pair(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """
    Mask the value pair in the dataset.
    :param dataset: Dataset where the input_ids have the true values and the labels have the [MASK] token. (NOTE: This will be swapped on return, makes it simpler for random mlm)
    :param tokenizer: Model tokenizer.
    :return: Masked dataset with input_ids having the [MASK] and the labels having -100 except the true masked value.
    """
    mask_token_id = tokenizer.mask_token_id
    return dataset.map(
        lambda data: {
            # Swap the input_ids and labels fields.
            "input_ids": data["labels"],
            "labels": [
                token if mask_id == mask_token_id else -100
                for mask_id, token in zip(data["labels"], data["input_ids"])
            ],
        },
        num_proc=1,
    )


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

    match loader.model_type:
        case ModelType.ENCODER_DECODER:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=loader.tokenizer,
                return_tensors="pt",
            )
        case ModelType.ENCODER:
            if loader.model_settings.training_encoder_only_mask_values:
                # Create a labels tensor with -100 for all non-masked tokens positions
                training_data = mask_value_pair(training_data, loader.tokenizer)
                data_collator = DataCollatorForMaskedValueTokens(loader.tokenizer)
            else:
                # Remove the labels column as its handled by the data collator
                training_data = training_data.remove_columns(["labels"])
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=loader.tokenizer,
                    mlm=True,
                    mlm_probability=0.4,
                    return_tensors="pt",
                )
        case ModelType.DECODER:
            training_data = training_data.remove_columns(["labels"])
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=loader.tokenizer,
                mlm=False,
                return_tensors="pt",
            )
    training_data = training_data.train_test_split(test_size=0.1, seed=42)

    # Stop training early
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,
        # Minimum eval improvement
        early_stopping_threshold=0.01,
    )

    trainer = Trainer(
        model=loader.model,
        args=training_args,
        train_dataset=training_data["train"],
        eval_dataset=training_data["test"],
        processing_class=loader.tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
    )

    # Train the model.
    trainer.train()

    # Evaluate the model on the test set.
    eval_results = trainer.evaluate()
    print("Final evaluation results:", eval_results)

    # Save the fine-tuned model and tokenizer.
    loader.model.save_pretrained(output_dir)
    loader.tokenizer.save_pretrained(output_dir)

    # Push fined tuned model to HF
    if trainer.args.push_to_hub:
        print("Pushing to hub...")
        trainer.push_to_hub("test")
    else:
        print("No .env file found. Pushing to hub skipped.")


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

    # Resize the model's token embeddings to fit the new tokens.
    model_loader.model.resize_token_embeddings(len(model_loader.tokenizer))

    # NOTE: Fix for: "CUDA Assertion `t >= 0 && t < n_classes` failed" for the ltg encoder and encoder-decoder models
    # The Classifier does not get resized when calling model.resize_token_embeddings() so needs to be manually re-initialized
    if model_loader.model_type == ModelType.ENCODER:
        model_loader.model.classifier.__init__(
            model_loader.model.config,
            model_loader.model.embedding.word_embedding.weight,
        )
    elif model_loader.model_type == ModelType.ENCODER_DECODER:
        model_loader.model.classifier.__init__(model_loader.model.config)


def reinitialize_weights(module):
    """Simulate untrained model by reinitializing the weights of the model layers."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
        print(f"Reinitialized weights for {module.__class__.__name__}")


def train(model_type: ModelType, model_index: int, push_to_hub: bool = True) -> None:
    """
    Train the model using the provided dataset.
    :param model_type: Model type to train.
    """
    model_loader = ModelLoader(model_type, model_index, is_trained=False)
    dataset_dir = "data/corrected/"
    output_dir = f"trained/{str(model_loader.model_settings)}"
    print(f"Saving trained model to: {output_dir}")

    # Huggingface token and repo path
    env_config: dict = dotenv_values(".env")
    hf_token = env_config.get("HUGGINGFACE_SECRET_TOKEN", None)
    hf_username = env_config.get("HUGGINGFACE_USERNAME", None)

    # Replace the old repo slash to make a valid repo name
    hf_model_id = f"{hf_username}/{str(model_loader.model_settings).replace('/', '_')}"

    # Define training args
    simulated_batch_size = 32
    batch_size = model_loader.model_settings.training_batch_size
    gradient_accumulation_steps = max(1, simulated_batch_size // batch_size)
    training_args = TrainingArguments(
        hub_token=hf_token,
        hub_model_id=hf_model_id,
        hub_private_repo=True,
        push_to_hub=push_to_hub,
        output_dir=output_dir,
        num_train_epochs=model_loader.model_settings.training_num_epochs,
        learning_rate=model_loader.model_settings.training_learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,  # Mixed precision
        weight_decay=0.01,
        warmup_ratio=0.1,  # 10% warmup
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,  # Limit max gradient
        eval_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
    )

    # Load the dataset.
    mask_token = "null"
    if model_type == ModelType.ENCODER:
        mask_token = model_loader.tokenizer.mask_token
    dataset, enums = DatasetLoader(model_type, mask=mask_token).create_dataset(
        dataset_dir, include_enums=False
    )
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

    # Unload the model
    model_loader.unload_model()


if __name__ == "__main__":
    # Train all model types and sizes except gemma and qwen
    for m_type in ModelType:
        for i in range(len(MODELS_DICT[m_type])):
            # Don't fine-tune gemma and qwen
            if m_type == ModelType.DECODER and i >= 2:
                break
            train(m_type, i)
