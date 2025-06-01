# From Free-Text to Structured Data: Extracting Medical Information with Large Language Models

This project focuses on extracting structured medical data from pathology reports in the Norwegian colorectal cancer screening program.  
It uses the HuggingFace library to allow for loading, fine-tuning and sharing of models with encoder, decoder and encoder-decoder architectures. 

### Key Features
- **Automatic labeling** using zero-shot language models  
- **Fine-tuning** to train smaller models using the auto labeled dataset
- **Evaluation** with visualizations using manually labeled data
- **Web-based labeling interface** for domain experts

A live demo of the labeling tool is available [here](https://veggissss.github.io/FFT2SD/).

The published fine-tuned models used in this project can be found on HuggingFace [here](https://huggingface.co/collections/Veggissss/fft2sd-fine-tuned-models-682dd356b0ba237438b05ed1).

[![Python Unit Tests](https://github.com/Veggissss/FFT2SD/actions/workflows/pytest.yml/badge.svg)](https://github.com/Veggissss/FFT2SD/actions/workflows/pytest.yml)
[![Frontend Unit Tests](https://github.com/Veggissss/FFT2SD/actions/workflows/frontend-vitest.yml/badge.svg)](https://github.com/Veggissss/FFT2SD/actions/workflows/frontend-vitest.yml)
[![Deploy demo site using mock data to GitHub Pages](https://github.com/Veggissss/FFT2SD/actions/workflows/deploy-mock-frontend-pages.yml/badge.svg)](https://github.com/Veggissss/FFT2SD/actions/workflows/deploy-mock-frontend-pages.yml)

## Project Structure Overview
<details>
<summary>
Root
</summary>
<details>
<summary><b>data/</b>: Dataset labeling</summary>

* **corrected/**: The manually labeled data from *batch* in separated JSON form.
* **auto_labeled**: The remainder of the dataset, automatically marked by an LLM.
* **large_batch/**: The raw unlabeled dataset with a *labeled_ids.json* keeping track of what has been manually labeled/*corrected*.
* **example_batch/**: Small sample of pathology reports (missing 'klinisk' info).
* **test_data/**: The labeled test data from *example_batch*, used for initial development.
* **label_data.py** Initial, simple cmd line program that takes user input and generates labeled data.
* **convert_dataset.py** Simple script that converts folder of JSONs to a JSONL file.
</details>

<details>
<summary><b>data_model/</b>: Data Model creation based on the <a href="/data_model/strukturert-rekvisisjon-og-svarrapport-for-patologirapportering-0.76.xlsx">pathology spreadsheet</a></summary>

* **generate_data_model.py**: Fill in each model *struct* where every entry gets a *null* "value" field and enum definitions are filled in using the enum.
* **struct/**: Containing all the base field for every report type and metadata with reference strings to enums such as *"REF_ENUM;Lokasjon"*.
* **enum/**: Contains all enum definitions.
* **figure/**: Combining the full data model and representing it as a more readable svg figure.
* **out/**: Output directory of the filled model *struct*
</details>

<details>
<summary><b>utils/</b>: Definitions and help functions.</summary>

- **enums.py**: Containing enum definitions mappings for model and report type.
- **token_constraints.py**: Generation constraints for stopping auto regressive models and finding allowed unmask tokens for the encoder model.
- **file_loader.py**: Helping functions for handling JSON and text files.
</details>

- **config.py**: Definitions of the used HuggingFace models. *Change the `MODELS_DICT` if you want to eval or train a different HuggingFace model.*
- **model_loader.py**: ModelLoader class for managing loading and generating output from the models.
- **model_strategy.py**: Containing model architecture specific logic for encoder, decoder and encoder-decoder models.
- **dataset_loader.py**: Loads the labeled JSON data, creates a prompt for every field in the JSON and stores it as a *HuggingFace Dataset*. Also finds all the unique enum definitions in the dataset which will be added as separate tokens.
- **model_train.py**: Training script that adds enum definitions as new tokens and trains/fine-tunes the models based on their architecture.
- **server.py**: Simple Flask API for using the models with POST requests. Request paths such as "/load_model" and "/generate".
- **tests/**: Folder containing *pytest* unit tests


</details>

<details>
<summary>
Libraries
</summary>
<ul>
  <li><strong>Backend</strong>
    <ul>
      <li><em>LLM:</em>
        <a href="https://pytorch.org/get-started/locally/" target="_blank">torch (cuda)</a>,
        <a href="https://pypi.org/project/transformers/" target="_blank">Transformers</a>,
        <a href="https://pypi.org/project/datasets/" target="_blank">datasets</a>,
        <a href="https://pypi.org/project/accelerate/" target="_blank">accelerate</a>,
        <a href="https://pypi.org/project/peft/" target="_blank">peft</a>
      </li>
      <li><em>API:</em> <a href="https://pypi.org/project/Flask/" target="_blank">Flask</a></li>
      <li><em>Testing:</em> <a href="https://pypi.org/project/pytest/" target="_blank">pytest</a></li>
    </ul>
  </li>
  <li><strong>Frontend</strong>
    <ul>
      <li><a href="https://react.dev/" target="_blank">React</a> + <a href="https://vitejs.dev/" target="_blank">Vite</a> + <a href="https://www.typescriptlang.org/" target="_blank">TypeScript</a></li>
      <li><a href="https://mswjs.io/" target="_blank">MSW</a> for API mocking</li>
    </ul>
  </li>
</ul>
</details>


## Installation

#### Prerequisites:
1. Install [Python](https://www.python.org/downloads/)
2. Install [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
    - Unix: `curl https://sh.rustup.rs -sSf | sh
`
    - Win: https://win.rustup.rs/

Setup virtual env:
1. `python -m venv venv`

    a. `venv\Scripts\activate` for Windows
    
    b. `source venv/bin/activate` for Unix

Install dependencies:

2. `pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple`

    Newer/other PyTorch CUDA versions: `https://pytorch.org/get-started/locally/`

#### Web app:

Prerequisites:
- [Node.js](https://nodejs.org/en)

Configuration:
- If hosting the API server on a seperate machine than the Web App, add an environment file `Frontend/.env`, with value:
  VITE_API_BASE_URL=http(s)://API_SERVER_ADDRESS:PORT_NUMBER


Install dependencies:
- `cd Frontend`
- `npm install`

Host dev:
- `npm run dev`
- Run `python server.py` for LLM API Backend


## How to Auto Label, Train and Evaluate:

#### Configuration:
To use models from HuggingFace which require accepting terms of service, are private, or to publish the fine-tuned models, you need to create a `.env` file containing your HuggingFace username and an [access token](https://huggingface.co/settings/tokens). 

See [`example.env`](example.env) for the correct format.

### Auto Script

Use [train_and_eval.sh](train_and_eval.sh) to install requirements with CUDA Torch. It will then Auto Label, Train and Evaluate all the models.

#### Auto Label
Use an LLM to auto label the remainder of the dataset:
`python model_auto_label.py`

You can change the model name
The generated training dataset will be in `data/auto_labeled/`

#### Train
To train all the models marked as `is_fine_tuning` set to `True` and use the Auto labeled dataset as training data.


#### Eval
Will use the manually labeled dataset to evaluate all the models and create [eval_results.json](eval_results.json) and figures placed in `figures/eval` 


## Structured Data Model
The data model is based upon the [strukturert-rekvisisjon-og-svarrapport-for-patologirapportering-0.76.xlsx](data_model/strukturert-rekvisisjon-og-svarrapport-for-patologirapportering-0.76.xlsx) spreadsheet. 
Info of the different fields can be found inside the spreadsheet as well as [Here](https://www.kreftregisteret.no/screening/tarmscreening/for-helsepersonell/kvalitetsmanual/kapittel-11-laboratorieprosedyre-for-patologitjenesten).

Since many of these required fields use predefined enum values they are being defined globally and by using "references" inside the JSON, the python script replaces the enum with its possible values. This allows for reuse as well as more readability.
The final generated data models can be found in the `model/out` folder. 
Resulting in data models for clinical, macroscopic and microscopic analysis reports.

The generated data models have a "value" field which is set to null. This is going to be given to the LLM along with the prose text for each reporting step (clinical, macroscopic, microscopic).

## UML Diagram of Structured Model
The figure is generated using [omute.net](https://omute.net/editor) [[git]](https://github.com/AykutSarac/jsoncrack.com).
This is a simplified figure where some enum values are grouped and "id" and "value": null is removed for readability.
![UML of Structured Data Model](data_model/figure/data_model_figure.svg)

## Data Collection Process
![Data Collection](figures/LLM.Overview.drawio.svg)

## Data Labeling
The labeling program takes in a prose text containing either *klinisk*, *makroskopisk* or *microskopisk* text.
The text content determines which of the data model for the different report types should be used.
The text is then mapped to the corresponding data model, in a one to many relationship. 
As seen in the example text below, a single report can contain multiple glass samples with different attributes, hence requiring a multiplicity relationship:
```js
//`data/example_batch/case_1_makro.txt`:
3 glass, merket 1 - 3
 1: 4  gryn i #1
 2: 3  gryn i #2
 3: 7  gryn i #3
```
Every field for the respective data model type is then prompted to be filled out by the user based on the glass number.
The metadata contains the report type, the total glass number in the text and the glass number of which is currently being filled out.

The resulting labeled data is then stored in a JSON with the fields:
- *input_text* | The original information text.
- *target_json* | The "correctly" labeled out JSON that the model will try to replicate. 
- *metadata_json* | The report type, total amount of glass containers in the original *input_text* as well as the number corresponding to the filled out *target_json*. (1-indexed)

![Labeling program flow](figures/Labeling.drawio.svg)

## Dataset Creation
One thing to keep in mind is that the LLMs used have a limited context window sizes and max lengths. 
Since some of the data models, namely the microscopic data model contain a lot of different enum values.
Strategies to mitigate this include removing non-critical fields such as "is_required" from the model and splitting up each prompts into multiple.
Other models with larger context length might be used to prevent this issue, but with higher resource usage since attention generally scales quadratically with sequence length.
Furthermore, the availability of such open source models and which are pre-trained on norwegian makes it out of scope for this project.

After using improved test data the spitting of prompts seems necessary. 
Now each JSON entry in the model is prompted individually instead of the whole model template per prompt.
This increases initial dataset creation complexity, but allows for more flexibility and allows for injecting and requesting specific fields.
This feature was later used to inject metadata info into the training data for allowing to extract things like report type and total glass containers.


## Training / Fine-tuning Process
The project will focus on using open source models that already have been trained to understand natural language. Since this project involves analyzing medical journal texts which contains prose written in norwegian, more specialized trained models will be used. Namely some of the norwegian trained models by the [Language Technology Group (University of Oslo) on HuggingFace](https://huggingface.co/ltg).

These models will be fine-tuned to extract medical information from the medical prose text and fill out the `null` fields from a JSON formatted data model. 
The models result will be compared with a pre-filled correct labeled data model corresponding to the given input.
The model will then use back-propagation to adjust its weights by using the calculated loss.

An overview of how the dataset is structured, along with training and the evaluation process:

![Training overview](figures/LLM.DataFlow.drawio.svg)

### Tokenization
Just before the training/fine-tuning process begins, new tokens from the dataset are added to the tokenizer.
The added tokens are enum values that can be used by the model to fill out the JSON.
By making each enum a single word token it makes it possible for the encoder model to unmask the value without having the token being split up into multiple tokens.
It also simplifies the process of allowing certain tokens as the `enum` field that specifies which values are allowed in the `value` field since each token will have a single dedicated id.
The model's embeddings (and classifier manual fix for ltg encoder/encoder-decoder) are resized to fit the new tokens.

### Restricted tokens
By reducing the possible allowed tokens the model can produce, the chances that an invalid token is generated is reduced.
This is done by setting the score values for the non allowed tokens to -inf, when unmasking for encoder models and by using a `LogitsProcessor` for autoregressive models (encoder-decoder & decoder). 
This `LogitsProcessor` triggers when the `value` field, a `:` and a `"` is generated.
The constrainted `LogitsProcessor` also force closes the `"` and generates a `}` which will stop the generation due to the `StoppingCriteria`. (See `token_constraints.py` for full implementation)

