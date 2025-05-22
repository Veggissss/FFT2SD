from pathlib import Path
from utils.file_loader import save_json, load_json
from utils.enums import ReportType, DatasetField
from utils.data_classes import TokenOptions
from model_loader import ModelLoader
import server

# Testing script to fully label the complete unlabeled dataset
# This could be used to quickly label the entire dataset using an (hopefully accurate) large model and use the dataset to fine tune smaller models
if __name__ == "__main__":
    server.model_loader = ModelLoader(
        is_trained=False,
        # Must match one of the models in config.py. (With included suffixes if quantized):
        load_model_name="google/gemma-3-4b-it",
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
            if field.value not in unlabeled or unlabeled[field.value] is None:
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
                # Save vram, but slow, set to None for faster generation:
                batch_size=1,
            )
            if not reports:
                print(f"No reports generated for input: {input_text}")
                continue
            print(f"Generated {len(reports)} reports for input: {input_text}")

            for i, report in enumerate(reports):
                # Save the generated report to a file
                save_json(
                    report,
                    output_path.joinpath(f"{report_type.value}_{report_id}_{i+1}.json"),
                )
                print(f"Saved report for {report_type.value} with ID {report_id}")
