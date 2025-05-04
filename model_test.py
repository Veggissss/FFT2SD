from pathlib import Path
from utils.file_loader import save_json, load_json
from utils.enums import ModelType, ReportType, DatasetField
from utils.data_classes import TokenOptions
from model_loader import ModelLoader
import server

# Testing script to fully label the complete unlabeled dataset
# This could be used to quickly label the entire dataset using an (hopefully accurate) large model and use the dataset to fine tune smaller models
if __name__ == "__main__":
    model_type = ModelType.ENCODER_DECODER
    model_index = 0
    is_trained = False
    server.model_loader = ModelLoader(model_type, model_index, is_trained)

    output_path = Path("data/test_label_all/")
    unlabeled_dataset = load_json("data/large_batch/export_2025-03-17.json")

    # Map text to correct data model
    dataset_field_mapping = {
        DatasetField.KLINISK: ReportType.KLINISK,
        DatasetField.MAKROSKOPISK: ReportType.MAKROSKOPISK,
        DatasetField.MIKROSKOPISK: ReportType.MIKROSKOPISK,
        DatasetField.DIAGNOSE: ReportType.MIKROSKOPISK,
    }

    for unlabeled in unlabeled_dataset:
        for field in DatasetField:
            if field.value not in unlabeled or field.value is None:
                continue

            report_id = unlabeled.get("id")
            report_type = dataset_field_mapping.get(field)
            token_options = TokenOptions(report_type=report_type)
            input_text = unlabeled[field.value]
            reports = server.generate(input_text, report_type=report_type)
            if not reports:
                print(f"No reports generated for input: {input_text}")
                continue

            for report in reports:
                # Save the generated report to a file
                save_json(
                    report, output_path.joinpath(f"{field.value}_{report_id}.json")
                )
