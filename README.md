# From Free-Text to Structured Data: Extracting Medical Information with Large Language Models

master-project.wip.exe.zip.lib.rar

## Project Structure Overview
- **model_loader.py**: ModelLoader class for managing loading and generating output from the models.
- **model_strategy.py**: Containing model architecture specific logic for encoder, decoder and encoder-decoder models.
- **dataset_loader.py**: Loads the labeled JSON data, creates a prompt for every field in the JSON and stores it as a *HuggingFace Dataset*. Also finds all the unique enum definitions in the dataset which will be added as separate tokens.
- **model_train.py**: Training script that adds enum definitions as new tokens and trains/fine-tunes the models based on their architecture.
- **server.py**: Simple Flask API for using the models with POST requests. Request paths for "/load_model" and "/generate".
- **tests/**: Folder containing *pytest* unit tests
<details>
<summary><b>data/</b>: Dataset labeling</summary>

* **batch/**: The full unlabeled text data real world pathology reports. ***##WIP##***
* **labeled_data/**: The fully labeled data from *batch* in separated json form. ***##WIP##***
* **example_batch/**: Small sample of pathology reports (missing 'klinisk' info).
* **test_data/**: The labeled test data from *example_batch*, used for initial development.
* **label_data.py** Simple data labeling program that takes user input and generates training data.
</details>

<details>
<summary><b>data_model/</b>: Data Model creation based on the <a href="#data-model">pathology spreadsheet</a></summary>

* **generate_data_model.py**: Fill in each model *struct* where every entry gets a *null* "value" field and enum definitions are filled in using the enum.
* **struct/**: Containing all the base field for every report type and metadata with reference strings to enums such as *"REF_ENUM;Lokasjon"*.
* **enum/**: Contains all enum definitions.
* **figure/**: Combining the full data model and representing it as a more readable svg figure.
* **out/**: Output directory of the filled model *struct*
</details>

<details>
<summary><b>utils/</b>: Decoupled configurations, definitions and help functions.</summary>

- **config.py**: Constant definitions, including the definition of the used HuggingFace models. *Change the MODELS_DICT if you want to train a different HuggingFace model.*
- **enums.py**: Containing enum definitions mappings for model and report type.
- **token_constraints.py**: Generation constraints for stopping auto regressive models and finding allowed unmask tokens for the encoder model.
- **file_loader.py**: Helping functions for handling json and text files.
</details>


## Data Model
The data model is based upon the [strukturert-rekvisisjon-og-svarrapport-for-patologirapportering-0.76.xlsx](https://www.kreftregisteret.no/globalassets/tarmkreftscreening/dokumenter/kvalitetsmanualen/vedlegg/strukturert-rekvisisjon-og-svarrapport-for-patologirapportering-0.76.xlsx) spreadsheet. 
Info of the different fields can be found inside the spreadsheet as well as [Here](https://www.kreftregisteret.no/screening/tarmscreening/for-helsepersonell/kvalitetsmanual/kapittel-11-laboratorieprosedyre-for-patologitjenesten).

Since many of these required fields use predefined enum values they are being defined globally and by using "references" inside the json, the python script replaces the enum with its possible values. This allows for reuse as well as more readability.
The final generated data models can be found in the `model/out` folder. 
Resulting in data models for clinical, macroscopic and microscopic analysis reports.

The generated data models have a "value" field which is set to null. This is going to be given to the LLM along with the prose text for each reporting step (clinical, macroscopic, microscopic).

~~**One thing to keep in mind is that the LLMs have different and a limited context window sizes and max lengths. Since some of the data models contain a lot of different enum values and/or fields that are required. Strategies to midigate this might include splitting up into multiple prompts/evals. Further investigation will occur once the extraction process of real data starts.**~~
Update: After using improved test data the spitting of prompts seems necessary. Now each json entry in the model is prompted individually instead of the whole model per prompt. This increases generation times, but is necessary for the output to not get cut off which due to the relatively small max length of the used models (~512 tokens). Other models with larger context length might be used to prevent this issue, but availability of such open source models and which are pre-trained on norwegian makes it out of scope for this project.

## UML Diagram of Structured Model
Generated figure using [omute.net](https://omute.net/editor) [[git]](https://github.com/AykutSarac/jsoncrack.com):
This is a simplified figure where some enum values are grouped and "id" and "value": null is removed for readability.
![UML of Structured Data Model](data_model/figure/data_model_figure.svg)

## Data Collection Process
Data extraction will start after approval, early 2025.

![Data Collection](figures/LLM.Overview.drawio.svg)

## Data Labeling
The labeling program takes in a text containing either *klinisk*, *makroskopisk* or *microskopisk* information. The text is then mapped to the corresponding data model. Every field from the data model is then prompted to be labeled by the user. In addition to the clinical information the user is prompted to fill in the data model based on the glass container. Since each text can include multiple containers that needs to be mapped in a one to many relation to the corresponding data model. 

The resulting labeled data is then stored in a json with the fields:
- *input_text* | The original information text.
- *target_json* | The "correctly" labeled out json that the model will try to replicate. 
- *metadata_json* | Info about the total amount of glass containers in the original *input_text* as well as the number corresponding to the filled out *target_json*. (1-indexed)

## Training / Fine-tuning Process
The project will focus on using open source models that already have been trained to understand natural language. Since this project involves analyzing medical journal texts which contains prose written in norwegian, more specialiced trained models will be used. Namely some of the norwegian trained models by the [Language Technology Group (University of Oslo) on HuggingFace](https://huggingface.co/ltg).

These models will be fine-tuned to extract medical information from the medical prose text and fill out the `null` fields from a json formatted data model. The models result will be compared with a pre-filled correct labeled data model corresponding to the given input.
The model will then use back-propagation to adjust its weights by using a loss function.

An overview of how the dataset is structured, along with training and the evaluation process:

![Training overview](figures/LLM.DataFlow.drawio.svg)

### Tokenization
- Added enums to tokenizer.

### Encoder 
- Masked learning
- Allowed tokens filtering
- *Single token restriction*
- Differ from NER

### Decoder
- Casual learning
- Prompt engineering; Starter tokens
- Masked attention

### Encoder-Decoder
- Sequence to Sequence

## Evaluation of Models
- Separate from training
- TODO: Evaluation metrics presented as graphical, tables etc.

## Investigate
- See if end of sentence marker and prompt impacts the different models.
- Encoder: 
    * Extra label specific masked training
    * One mask to many tokens. (String comments etc)
- Create simple frontend UI to illustrate the use of the server.py API.

## Libraries
- pytest: Unit testing