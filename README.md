# From Free-Text to Structured Data: Extracting Medical Information with Large Language Models

master-project.wip.exe.zip.lib.rar

## Data Model
JSON is going to be used by the LLM.

The data model is based upon the [strukturert-rekvisisjon-og-svarrapport-for-patologirapportering-0.76.xlsx](https://www.kreftregisteret.no/globalassets/tarmkreftscreening/dokumenter/kvalitetsmanualen/vedlegg/strukturert-rekvisisjon-og-svarrapport-for-patologirapportering-0.76.xlsx). Info of the different fields can be found [Here](https://www.kreftregisteret.no/screening/tarmscreening/for-helsepersonell/kvalitetsmanual/kapittel-11-laboratorieprosedyre-for-patologitjenesten)!

Since many of these required fields use predefined enum values they are being defined globally and by using "references" inside the json the python script replaces the enum with the possible values. This allows for reuse as well as more readability.

