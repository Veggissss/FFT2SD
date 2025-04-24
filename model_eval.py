from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

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
):
    """
    Runs evaluation on all labeled JSON files
    Saves per-file metrics to a JSON file for each model_type.
    """
    server.model_loader = ModelLoader(model_type, model_index, is_trained)

    files = list(data_dir.glob("*.json"))
    per_file_stats = []

    for file_path in files:
        data = load_json(file_path)
        input_text = data["input_text"]
        target = data["target_json"]
        metadata = data["metadata_json"]
        report_type = ReportType(metadata[0]["value"])

        response = server.generate(input_text, report_type, total_containers=1)
        predicted = response[0].get("target_json", [])

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
):
    """
    Loads evaluation results and visualizes:
    - Accuracy by Report Type and Model Type
    - Accuracy by Value Type and Model Type
    - Precision / Recall / F1 by Model Type
    """
    results = load_json(results_path)
    if ignore_strings:
        results = [entry for entry in results if entry.get("type") != "string"]

    # Filter out other model sizes
    results = [
        entry for entry in results if entry.get("model_type") == model_type.value
    ]
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


if __name__ == "__main__":
    for m_type in ModelType:
        # Test with small models
        break
        evaluate(
            model_type=m_type,
            model_index=0,
            is_trained=True,
        )

        continue
        for i in range(len(MODELS_DICT[m_type])):
            is_trained = m_type != ModelType.DECODER or i < 3
            evaluate(
                model_type=m_type,
                model_index=i,
                is_trained=is_trained,  # Only gemma and deepseek are not trained
            )

    for m_type in ModelType:
        visualize(model_type=m_type, ignore_strings=True)
