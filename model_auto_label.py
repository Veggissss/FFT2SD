from pathlib import Path
from utils.file_loader import save_json, load_json
from utils.enums import ModelType, ReportType, DatasetField
from utils.data_classes import TokenOptions
from model_loader import ModelLoader
import server

# Testing script to fully label the complete unlabeled dataset
# This could be used to quickly label the entire dataset using an (hopefully accurate) large model and use the dataset to fine tune smaller models
if __name__ == "__main__":
    BASE_MODEL_NAME = "google/gemma-3-27b-it"

    server.model_loader = ModelLoader(
        model_type=ModelType.DECODER, is_trained=False, base_model_name=BASE_MODEL_NAME
    )

    output_path = Path("data/auto_labeled/")
    unlabeled_dataset = load_json("data/large_batch/export_2025-03-17.json")
    labeled_ids = load_json("data/large_batch/labeled_ids.json")

    # Map text to correct data model
    dataset_field_mapping = {
        DatasetField.KLINISK: ReportType.KLINISK,
        DatasetField.MAKROSKOPISK: ReportType.MAKROSKOPISK,
        DatasetField.MIKROSKOPISK: ReportType.MIKROSKOPISK,
        DatasetField.DIAGNOSE: ReportType.MIKROSKOPISK,
    }

    for unlabeled in unlabeled_dataset:
        # Prevent labeling same as "ground truth"
        report_id = unlabeled.get("id")
        if report_id in labeled_ids:
            print(f"Report ID {report_id} already labeled, skipping.")
            continue

        # Use text for every field in the dataset
        for field in DatasetField:
            if field.value not in unlabeled or field.value is None:
                continue

            # Set token generation options
            report_type = dataset_field_mapping.get(field)
            token_options = TokenOptions(report_type=report_type)
            token_options.include_enums = True  # Include enum values in the prompt
            # token_options.generate_strings = True  # Generate string values in the output

            input_text = unlabeled[field.value]
            reports = server.generate(
                input_text,
                report_type=report_type,
                token_options=token_options,
                allow_metadata_null=True,
            )
            if not reports:
                print(f"No reports generated for input: {input_text}")
                continue
            print(f"Generated {len(reports)} reports for input: {input_text}")

            for report in reports:
                # Save the generated report to a file
                save_json(
                    report, output_path.joinpath(f"{field.value}_{report_id}.json")
                )
                print(f"Saved report for {field.value} with ID {report_id}")
