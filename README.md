# From Free-Text to Structured Data: Extracting Medical Information with Large Language Models

master-project.wip.exe.zip.lib.rar

## Data Model
The data model is based upon the [strukturert-rekvisisjon-og-svarrapport-for-patologirapportering-0.76.xlsx](https://www.kreftregisteret.no/globalassets/tarmkreftscreening/dokumenter/kvalitetsmanualen/vedlegg/strukturert-rekvisisjon-og-svarrapport-for-patologirapportering-0.76.xlsx) spreadsheet. 
Info of the different fields can be found inside the spreadsheet as well as [Here](https://www.kreftregisteret.no/screening/tarmscreening/for-helsepersonell/kvalitetsmanual/kapittel-11-laboratorieprosedyre-for-patologitjenesten).

Since many of these required fields use predefined enum values they are being defined globally and by using "references" inside the json, the python script replaces the enum with its possible values. This allows for reuse as well as more readability.
The final generated datamodels can be found in the `model/out` folder. 
Resulting in datamodels for clinical, macroscopic and microscopic analysis reports.

The generated datamodels have a "value" field which is set to null. This is going to be given to the LLM along with the prose text for each reporting step (clinical, macroscopic, microscopic).

**One thing to keep in mind is that the LLMs have different and a limited context window sizes and max lengths. Since some of the datamodels contain a lot of different enum values and/or fields that are required. Strategies to midigate this might include splitting up into multiple prompts/evals. Further investigation will occur once the extraction prosess of real data starts.**


## Data Collection Process
Data extraction will start after approval, early 2025.

![Data Collection](figures/LLM.Overview.drawio.svg)


## Training / Fine-tuning process
The project will focus on using open source models that already have been trained to understand natural language. Since this project involves analyzing medical journal texts which contains prose written in norwegian, more specialiced trained models will be used. Namely some of the norwegian trained models by the [Language Technology Group (University of Oslo) on HuggingFace](https://huggingface.co/ltg).

These models will be fine-tuned to extract medical information from the medical prose text and fill out the `null` fields from a json formatted datamodel. The models result will be compared with a pre-filled correct labeled datamodel corresponding to the given input.
The model will then use backpropegation to adjust its weights by using a loss function.

An overview of how the dataset is structured, along with training and the evaluation process:

![Training overview](figures/LLM.DataFlow.drawio.svg)


