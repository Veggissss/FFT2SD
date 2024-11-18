from transformers import (
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import json

from model_loader import ModelLoader
from config import MODELS_DICT, SYSTEM_PROMPT


def preprocess_function(examples: dict, model_loader: ModelLoader, max_length=512):
    """
    Prepare the dataset for training by creating input-output pairs.
    :param examples: Batch of examples containing 'text', 'json_template', and 'target_json'.
    :param model_loader: ModelLoader object with the loaded model and tokenizer.
    :return: Tokenized input and output tensors.
    """
    inputs = []
    targets = []

    for text, json_template, target_json in zip(
        examples["input_text"], examples["json_template"], examples["target_json"]
    ):
        if model_type in ["encoder-decoder", "decoder"]:
            # Combine text and JSON template into a prompt.
            input_text = SYSTEM_PROMPT.format(
                input_text=text,
                template_str=json_template,
                prompt_separator="<END_OF_PROMPT>",
            )
            target_text = target_json
        elif model_type == "encoder":
            # Input is just the text, target might need to be custom-processed for training.
            input_text = text
            target_text = target_json

        inputs.append(input_text)
        targets.append(target_text)

    # Tokenize inputs and targets.
    model_inputs = model_loader.tokenizer(
        inputs,  # "input_ids" column in dataset
        text_target=targets,  # "labels" column in dataset
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    ).to(model_loader.device)

    return model_inputs


def load_and_prepare_data(model_loader: ModelLoader, max_length=512):
    """
    Load and prepare the dataset for training.
    :param model_loader: ModelLoader object with the loaded model and tokenizer.
    :param max_length: Maximum length of the input sequence.
    :return: Prepared dataset.
    """
    # Test dataset TODO: load real data from .jsonl
    data = {
        "input_text": ["The car is a 2021 model with 150 HP and comes in red color."],
        "json_template": [
            json.dumps(
                [
                    {"id": 1, "field": "model_year", "value": None, "required": True},
                    {"id": 2, "field": "horsepower", "value": None, "required": True},
                    {"id": 3, "field": "color", "value": None, "required": True},
                ]
            ),
        ],
        "target_json": [
            json.dumps(
                [
                    {"id": 1, "field": "model_year", "value": 2021, "required": True},
                    {"id": 2, "field": "horsepower", "value": 150, "required": True},
                    {"id": 3, "field": "color", "value": "red", "required": True},
                ]
            ),
        ],
    }

    # Convert dict to Hugging Face Dataset.
    dataset = Dataset.from_dict(data)

    return dataset.map(
        lambda x: preprocess_function(x, model_loader, max_length),
        batched=True,
        batch_size=4,
    )


def train_model(
    model_loader: ModelLoader, dataset: Dataset, output_dir="fine_tuned_model"
):
    """
    Train the model using the provided dataset.
    :param model: The model to be fine-tuned.
    :param tokenizer: Corresponding tokenizer.
    :param dataset: Preprocessed dataset.
    :param output_dir: Directory to save the fine-tuned model.
    """
    # Define training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,  # TODO: Set to a unique path for each model
        eval_strategy="no",  # TODO: Set to "epoch"
        learning_rate=2e-5,
        # per_device_train_batch_size=4,
        num_train_epochs=50,  # TODO Make selectable along with other training params
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize the Trainer.
    trainer = Trainer(
        model=model_loader.model,
        args=training_args,
        train_dataset=dataset,
        processing_class=model_loader.tokenizer,
        # eval_dataset= # TODO: Add evaluation dataset and set eval_strategy to "epoch"
        # data_collator=
    )

    # Train the model.
    trainer.train()

    # Save the fine-tuned model.
    model_loader.model.save_pretrained(output_dir)
    model_loader.tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    model_type = "encoder-decoder"

    # Load model and tokenizer.
    model_loader = ModelLoader(MODELS_DICT[model_type], model_type)

    # Load and preprocess the dataset.
    dataset = load_and_prepare_data(model_loader)

    # Fine-tune the model.
    train_model(model_loader, dataset)
