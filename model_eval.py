from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.data_classes import TokenOptions
from utils.file_loader import load_json, save_json
from utils.enums import ModelType, ReportType
from model_loader import ModelLoader
from config import MODELS_DICT
import server


def evaluate(
    model_type: ModelType,
    model_index: int = 0,
    is_trained: bool = True,
    data_dir: Path = Path("data/corrected"),
    output_path: Path = Path("./eval_results.json"),
    token_options: TokenOptions = None,
):
    """
    Runs evaluation on all labeled JSON files
    Saves per-file metrics to a JSON file for each model_type.
    """
    if server.model_loader:
        server.model_loader.unload_model()
    server.model_loader = ModelLoader(model_type, model_index, is_trained)

    files = list(data_dir.glob("*.json"))
    per_file_stats = []

    for file_path in files:
        data = load_json(file_path)
        input_text = data["input_text"]
        target = data["target_json"]
        metadata = data["metadata_json"]
        report_type = ReportType(metadata[0]["value"])
        container_id = str(metadata[2]["value"])

        generated_report = server.generate_container(
            input_text, report_type, container_id, metadata, token_options
        )
        predicted = generated_report.get("target_json", [])

        for target_item, predicted_item in zip(target, predicted):
            value_type = target_item.get("type", None)
            y_true = target_item.get("value")
            y_pred = predicted_item.get("value")
            correct = y_true == y_pred

            per_file_stats.append(
                {
                    "file": file_path.name,
                    "report_type": report_type.name,
                    "model_name": server.model_loader.model_name,
                    "model_type": model_type.value,
                    "type": value_type,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "accuracy": int(correct),
                }
            )

    # Overwrite previous results from the same model type
    if output_path.exists():
        existing = load_json(output_path)
        if isinstance(existing, list):
            existing = [
                entry
                for entry in existing
                if entry.get("model_name") != server.model_loader.model_name
            ]
            per_file_stats = existing + per_file_stats

    save_json(per_file_stats, str(output_path))
    print(f"Per-file metrics saved to {output_path}")


def add_bar_labels(ax):
    """Add numbered labels to bar plots."""
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )


def visualize(
    model_type: ModelType,
    output_dir: Path = Path("./figures/eval"),
    results_path: Path = Path("./eval_results.json"),
    ignore_strings: bool = True,
    ignore_null: bool = True,
    included_model_names: list[str] = None,
):
    """
    Loads evaluation results and visualizes:
    - Accuracy by Report Type and Model Type
    - Accuracy by Value Type and Model Type
    - Precision / Recall / F1 by Model Type
    """
    results = load_json(results_path)
    # Filter strings and null if enabled
    if ignore_strings:
        results = [entry for entry in results if entry.get("type") != "string"]
    if ignore_null:
        results = [entry for entry in results if entry.get("y_true") != "null"]
    # Filter out other model types
    results = [
        entry for entry in results if entry.get("model_type") == model_type.value
    ]
    # Filter out other model names if specified
    if included_model_names:
        results = [
            entry
            for entry in results
            if entry.get("model_name") in included_model_names
        ]

    if not results:
        print(f"No results found for {model_type.value}:\n{included_model_names}")
        return

    df = pd.DataFrame(results)

    # Accuracy by Report Type
    report_summary = (
        df.groupby(["report_type", "model_name"])["accuracy"].mean().unstack()
    )
    ax = report_summary.plot(kind="bar", figsize=(12, 6))
    add_bar_labels(ax)
    plt.title(f"Accuracy by Report Type ({model_type.value})")
    plt.xlabel("Report Type")
    plt.ylabel("Average Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title="Model Type")
    plt.tight_layout()
    plt.savefig(output_dir.joinpath(f"accuracy_by_report_type_{model_type.value}.svg"))
    print(f"Saved: accuracy_by_report_type_{model_type.value}.svg")

    # Accuracy by Value Type
    type_summary = df.groupby(["type", "model_name"])["accuracy"].mean().unstack()
    ax = type_summary.plot(kind="bar", figsize=(12, 6))
    add_bar_labels(ax)
    plt.title(f"Accuracy by Value Type ({model_type.value})")
    plt.xlabel("Value Type")
    plt.ylabel("Average Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title="Model Name")
    plt.tight_layout()
    plt.savefig(output_dir.joinpath(f"accuracy_by_value_type_{model_type.value}.svg"))
    print(f"Saved: accuracy_by_value_type_{model_type.value}.svg")

    # Precision / Recall / F1 per Model Type
    metrics = []
    for model in df["model_name"].unique():
        model_df = df[df["model_name"] == model]
        y_true = model_df["y_true"].astype(str)
        y_pred = model_df["y_pred"].astype(str)

        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        metrics.append(
            {
                "model_name": model,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

    metrics_df = pd.DataFrame(metrics).set_index("model_name")
    ax = metrics_df.plot(kind="bar", figsize=(12, 6))
    add_bar_labels(ax)
    plt.title(f"Precision, Recall, and F1 Score per Model Type ({model_type.value})")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir.joinpath(f"precision_recall_f1_{model_type.value}.svg"))
    print(f"Saved: precision_recall_f1_{model_type.value}.svg")


def test_masked_value():
    """
    Small test to compare value only vs random mlm training.
    """
    evaluate(ModelType.ENCODER, 0, is_trained=True)
    evaluate(ModelType.ENCODER, 1, is_trained=True)
    visualize(
        model_type=ModelType.ENCODER,
        ignore_strings=True,
        included_model_names=[
            "trained/ltg/norbert3-small_mask_values",
            "trained/ltg/norbert3-small",
        ],
        output_dir=Path("./figures/eval/value_training"),
    )


if __name__ == "__main__":
    # test_masked_value()

    for m_type in ModelType:
        token_options = TokenOptions()
        token_options.include_enums = m_type == ModelType.DECODER
        # token_options.generate_strings = True

        # Test all models
        for i, model_setting in enumerate(MODELS_DICT[m_type]):
            # All decoder models are evaluated untrained (0-shot), the rest are trained
            evaluate(
                model_type=m_type,
                model_index=i,
                is_trained=m_type != ModelType.DECODER,
                token_options=token_options,
            )

            # Evaluate trained decoder norallm models as well
            if "norallm" in model_setting.model_name:
                evaluate(
                    model_type=m_type,
                    model_index=i,
                    is_trained=True,
                    token_options=token_options,
                )

    # Trained Encoder
    visualize(
        model_type=ModelType.ENCODER,
        ignore_strings=True,
        included_model_names=[
            "trained/ltg/norbert3-small",
            "trained/ltg/norbert3-base",
            "trained/ltg/norbert3-large",
        ],
        output_dir=Path("./figures/eval/encoder"),
    )

    # Trained Encoder-Decoder models
    visualize(
        model_type=ModelType.ENCODER_DECODER,
        ignore_strings=True,
        included_model_names=[
            "trained/ltg/nort5-small",
            "trained/ltg/nort5-base",
            "trained/ltg/nort5-large",
        ],
        output_dir=Path("./figures/eval/encoder_decoder"),
    )

    # Trained Small decoder models and all its variants tained and untrained with/without 4bit quantization
    visualize(
        model_type=ModelType.DECODER,
        ignore_strings=True,
        included_model_names=[
            "norallm/normistral-7b-warm_4bit_quant",
            "norallm/normistral-7b-warm",
            "trained/norallm/normistral-7b-warm_4bit_quant",
            "trained/norallm/normistral-7b-warm",
        ],
        output_dir=Path("./figures/eval/decoder"),
    )

    # 0 shot test for larger untrained decoder models
    visualize(
        model_type=ModelType.DECODER,
        ignore_strings=True,
        included_model_names=[
            "norallm/normistral-7b-warm",
            "google/gemma-3-27b-it",
            "Qwen/Qwen3-32B",
        ],
        output_dir=Path("./figures/eval/0_shot"),
    )
