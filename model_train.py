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
import os


def load_and_prepare_data(model_loader: ModelLoader, dataset_path, max_length=512):
    """
    Load and prepare the dataset for training.
    :param model_loader: ModelLoader object with the loaded model and tokenizer.
    :param max_length: Maximum length of the tensors.
    :return: Prepared dataset.
    """
    # Dataset dictionary
    new_data = {
        "input": [],
        "output": [],
    }
    for filename in os.listdir(dataset_path):
        if not filename.endswith(".json"):
            print(f"Skipping non-JSON file: {filename}")
            continue

        # Load JSON file and process data
        with open(os.path.join(dataset_path, filename), "r", encoding="utf-8") as f:
            loaded_json_data = json.load(f)

            text = json.dumps(loaded_json_data["input_text"])
            template_json = json.loads(json.dumps(loaded_json_data["template_json"]))
            target_json = json.loads(json.dumps(loaded_json_data["target_json"]))

            # Iterate through template and target JSON entries
            for template_entry, target_entry in zip(template_json, target_json):
                template_entry_str = json.dumps(template_entry)
                target_entry_str = json.dumps(target_entry)

                if model_loader.model_type == "encoder":
                    template_entry_str = template_entry_str.replace(
                        '"value": null', f'"value": {model_loader.tokenizer.mask_token}'
                    )

                input_text = SYSTEM_PROMPT.format(
                    input_text=text,
                    template_json=template_entry_str,
                )

                if model_loader.model_type in ["encoder", "decoder"]:
                    target_text = SYSTEM_PROMPT.format(
                        input_text=text,
                        template_json=target_entry_str,
                    )
                else:
                    target_text = target_entry_str + " " + END_OF_PROMPT_MARKER

                new_data["input"].append(input_text)
                new_data["output"].append(target_text)

    # Convert dict to Hugging Face Dataset.
    dataset = Dataset.from_dict(new_data)

    print(dataset["input"][0])
    print(dataset["output"][0])

    output_size = len(dataset["output"])
    print(f"Number of examples: {output_size}")

    dataset = dataset.map(
        lambda data: model_loader.tokenizer(
            data["input"],
            text_target=data["output"],
            padding=True,
            truncation=True,
            return_tensors="np",  # NumPy is faster: https://huggingface.co/docs/datasets/nlp_process#map
        ),
        batched=True,
    )

    return dataset


def train_model(model_loader: ModelLoader, dataset: Dataset, output_dir: str):
    """
    Train the model using the provided dataset.
    :param model_loader: ModelLoader object with the loaded model and tokenizer.
    :param dataset: Preprocessed dataset.
    :param output_dir: Directory to save the fine-tuned model.
    """
    # Define training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,  # TODO: Set to a unique path for each model
        eval_strategy="no",  # TODO: Set to "epoch"
        learning_rate=2e-5,
        num_train_epochs=20,  # TODO Make selectable along with other training params
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,  # Enable mixed precision
    )

    # dataset = dataset.remove_columns(["input", "output"])
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
    model_type = "encoder-decoder"

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
    dataset = load_and_prepare_data(model_loader, "data/labeled_data/test/")

    # Fine-tune the model.
    train_model(model_loader, dataset, f"trained/{model_type}")
