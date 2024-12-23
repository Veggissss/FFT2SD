from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import json
from model_loader import ModelLoader
from config import MODELS_DICT, SYSTEM_PROMPT, END_OF_PROMPT_MARKER


def preprocess_dataset(
    dataset: dict, model_loader: ModelLoader, max_length=512
) -> list:
    """
    Prepare the dataset for training by creating input-output pairs.
    :param examples: Batch of examples containing 'text', 'json_template', and 'target_json'.
    :param model_loader: ModelLoader object with the loaded model and tokenizer.
    :return: Tokenized input and output tensors.
    """

    # Get input and output list of size 1 and create a tensor mapped

    return model_loader.tokenizer(
        dataset["input"],
        text_target=dataset["output"],
        return_tensors="pt",
        # max_length=max_length,
        truncation=True,
        padding=True,
    ).to(model_loader.device)


def load_and_prepare_data(model_loader: ModelLoader, max_length=512):
    """
    Load and prepare the dataset for training.
    :param model_loader: ModelLoader object with the loaded model and tokenizer.
    :param max_length: Maximum length of the input sequence.
    :return: Prepared dataset.
    """

    # Load the json template for clinical data report
    with open("model/out/generated-klinisk.json", "r", encoding="utf-8") as f:
        template_json = json.load(f)

    # Load the input text and the correct output
    with open("data/test/test-klinisk.json", "r", encoding="utf-8") as f:
        test_json = json.load(f)

    # Test dataset TODO: load real data from .jsonl
    data = {
        "input_text": [json.dumps(test_json["input_text"])],
        "template_json": [json.dumps(template_json)],
        "target_json": [json.dumps(test_json["target_json"])],
    }

    text = data["input_text"]
    template_json = data["template_json"]
    target_json = data["target_json"]

    new_data = {
        "input": [],
        "output": [],
    }

    # TODO Verify with multiple entries
    # for each entry create a new input-output pair
    for [entry_index, _] in enumerate(template_json):
        for template_json_entry, target_json_entry in zip(
            json.loads(template_json[entry_index]), json.loads(target_json[entry_index])
        ):
            template_json_entry = json.dumps(template_json_entry)
            target_json_entry = json.dumps(target_json_entry)

            if model_loader.model_type == "encoder":
                template_json_entry = template_json_entry.replace(
                    '"value": null', f'"value": {model_loader.tokenizer.mask_token}'
                )

            if model_loader.model_type in ["encoder", "decoder"]:
                input_text = SYSTEM_PROMPT.format(
                    input_text=text[entry_index], template_json=template_json_entry
                )

                target_text = SYSTEM_PROMPT.format(
                    input_text=text[entry_index], template_json=target_json_entry
                )
            else:
                input_text = SYSTEM_PROMPT.format(
                    input_text=text[entry_index], template_json=template_json_entry
                )
                target_text = target_json_entry + END_OF_PROMPT_MARKER

            new_data["input"].append(input_text)
            new_data["output"].append(target_text)

    # Convert dict to Hugging Face Dataset.
    dataset = Dataset.from_dict(new_data)

    print(dataset["input"][0])
    print(dataset["output"][0])

    batch_size = len(dataset["input"])
    output_size = len(dataset["output"])
    print(f"Batch size: {batch_size}")
    print(f"Number of examples: {output_size}")

    return dataset.map(
        lambda x: preprocess_dataset(x, model_loader, max_length),
        batched=True,
        batch_size=batch_size,
    )


def train_model(model_loader: ModelLoader, dataset: Dataset, output_dir: str):
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
        num_train_epochs=8,  # TODO Make selectable along with other training params
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,  # Enable mixed precision
    )

    dataset = dataset.remove_columns(["input", "output"])
    # dataset = dataset.train_test_split(test_size=0.1)

    if model_loader.model_type == "encoder" or model_loader.model_type == "decoder":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=model_loader.tokenizer,
            mlm=False,
            return_tensors="pt",
        )

        trainer = Trainer(
            model=model_loader.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=model_loader.tokenizer,
            # eval_dataset= # TODO: Add evaluation dataset and set eval_strategy to "epoch"
            data_collator=data_collator,
        )

    else:
        # Initialize the Trainer.
        trainer = Trainer(
            model=model_loader.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=model_loader.tokenizer,
            # eval_dataset= # TODO: Add evaluation dataset and set eval_strategy to "epoch"
            # data_collator=data_collator,
        )

    # Train the model.
    trainer.train()

    # Save the fine-tuned model.
    model_loader.model.save_pretrained(output_dir)
    model_loader.tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    model_type = "decoder"

    # Load model and tokenizer.
    model_loader = ModelLoader(MODELS_DICT[model_type], model_type)

    if model_type == "decoder":
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

    # Load and preprocess the dataset.
    dataset = load_and_prepare_data(model_loader)

    # Fine-tune the model.
    train_model(model_loader, dataset, f"trained/{model_type}")
