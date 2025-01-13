import json
import os


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_json_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(file_path: str, data: dict) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def main(
    input_text_name: str,
    input_text_path: str,
    input_json_path: str,
    output_dir_name: str,
) -> None:
    input_text = read_text_file(input_text_path)
    target_json = read_json_file(input_json_path)

    print(f"Input text:\n{input_text}\n")

    report_count = input("How many containers/Beholder-IDs are there?\n: ")
    try:
        report_count = int(report_count)
        if report_count < 1:
            print("Invalid input. Please enter a positive integer.")
            return

        for i in range(report_count):
            # For every field in the JSON, fill out the value field
            for item in target_json:
                print(f"Input text:\n{input_text}")
                print(f"Container Number {i + 1}:")

                isValid = False
                while not isValid:
                    labeled_input = input(f"Enter value for \n{item}\n: ")

                    # If the field is an enum, check if the input is valid
                    if "enum" in item["type"]:
                        if labeled_input not in item["enum"]:
                            # Check if the input is an enum index
                            try:
                                input_index = int(labeled_input)
                                if input_index < 0 or input_index >= len(item["enum"]):
                                    print(
                                        f"Invalid input! Must be one of {item['enum']}"
                                    )
                                    continue
                                labeled_input = item["enum"][input_index]
                            except ValueError:
                                print(f"Invalid input! Must be one of {item['enum']}")
                                continue

                    # If the field is a number, check if the input is a number
                    elif "float" in item["type"]:
                        try:
                            labeled_input = float(labeled_input)
                        except ValueError:
                            print("Invalid input! Please enter a float.")
                            continue

                    elif "int" in item["type"]:
                        try:
                            labeled_input = int(labeled_input)
                        except ValueError:
                            print("Invalid input! Please enter an int.")
                            continue

                    # If the field is a string, check if the input is a string
                    elif "string" in item["type"]:
                        if not isinstance(labeled_input, str):
                            print("Invalid input! Please enter a string.")
                            continue

                    # If the field is a boolean, check if the input is a boolean
                    elif "boolean" in item["type"]:
                        if labeled_input.lower() not in ["true", "false"]:
                            print("Invalid input! Please enter 'true' or 'false'.")
                            continue

                        if labeled_input.lower() == "true":
                            labeled_input = True
                        else:
                            labeled_input = False

                    isValid = True

                # Update the JSON with the valid labeled input
                item["value"] = labeled_input

            # Combine input text and target JSON
            final_json = {"input_text": input_text, "target_json": target_json}
            print(
                f"JSON Labeled:\n{json.dumps(final_json, indent=4, ensure_ascii=False)}\n"
            )

            # Create output directory if it doesn't exist
            output_dir = os.path.join(script_dir, "labeled_data", output_dir_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save to file
            json_file_name = input_text_name.replace(".txt", ".json")
            output_json_path = f"{output_dir}/container_{i}_{json_file_name}"
            write_json_file(output_json_path, final_json)

    except ValueError:
        print("Invalid input. Please enter an integer.")
        return


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    # Dir with text files to be labeled
    input_text_dir = os.path.join(script_dir, "example_batch")

    # Generated structured json files from data_model/out/
    input_json_dir = os.path.join(script_dir, "../data_model/out")

    # Output directory name
    output_dir_name = os.path.join(script_dir, "test")

    # For every text case fill out the JSON values
    for text_file in os.listdir(input_text_dir):
        if text_file.endswith(".txt"):
            text_file_path = os.path.join(input_text_dir, text_file)

            if "klinisk" in text_file or "diagn" in text_file:
                json_file_path = os.path.join(input_json_dir, "generated-klinisk.json")

            elif "makro" in text_file:
                json_file_path = os.path.join(
                    input_json_dir, "generated-makroskopisk.json"
                )

            elif "mikro" in text_file:
                json_file_path = os.path.join(
                    input_json_dir, "generated-mikroskopisk.json"
                )

            main(text_file, text_file_path, json_file_path, output_dir_name)
