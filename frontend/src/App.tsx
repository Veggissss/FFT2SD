import { useState } from 'react';
import ModelPanel from './components/ModelPanel';
import InputPanel from './components/InputPanel';
import OutputPanel from './components/OutputPanel';
import useApi from './hooks/useApi';
import { JsonItem } from './types';
import './App.css';

function App() {
    const [inputText, setInputText] = useState('');
    const [outputText, setOutputText] = useState('');
    const [modelType, setModelType] = useState('decoder');
    const [selectedIndex, setSelectedIndex] = useState<number>(0);
    const [isTrained, setIsTrained] = useState<boolean>(false);

    const [reportType, setReportType] = useState('auto');
    const [totalContainers, setTotalContainers] = useState<number | null>(null);
    const [jsonList, setJsonList] = useState<JsonItem[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [reportId, setReportId] = useState<string | null>(null)
    const [useFormInput, setUseFormInput] = useState(false);
    const [includeEnums, setIncludeEnums] = useState(true);
    const [generateStrings, setGenerateStrings] = useState(false);

    // API hooks
    const { isLoading, loadModel, generateReport, submitCorrection, getUnlabeled } = useApi();

    const handleSetModelType = (type: string) => {
        if (type === "encoder") {
            setIsTrained(true);
            setGenerateStrings(false);
        }
        setIncludeEnums(type === "decoder");
        setModelType(type);
    }

    const handleIsTrained = (checked: boolean) => {
        setIsTrained(checked);
    }

    const handleLoadModel = async () => {
        try {
            const data = await loadModel(modelType, selectedIndex, isTrained);
            console.log(data);
        } catch (error) {
            alert('Error loading model. Please try again.');
        }
    };

    const handleGenerate = async () => {
        try {
            const reportTypeToUse = reportType === "diagnose" ? "mikroskopisk" : reportType;
            const data = await generateReport(inputText, reportTypeToUse, totalContainers, includeEnums, generateStrings);
            if (reportType === 'auto') {
                setJsonList([]);
            }

            // Filter out duplicate reports
            const filteredJsonList = jsonList.filter(item => item.input_text && item.input_text !== inputText);

            // Add new data to list, and update index to be end of old list and start of new.
            const combined = filteredJsonList.concat(data);
            const index = Math.min(filteredJsonList.length, combined.length);

            setJsonList(combined);
            setCurrentIndex(index);
            setOutputText(JSON.stringify(combined[index], null, 2));
        } catch (error) {
            alert('Error generating report. Please try again.');
            console.log(error);
        }
    };

    const handleNext = () => {
        if (currentIndex < jsonList.length - 1) {
            try {
                const updatedJsonList = [...jsonList];
                updatedJsonList[currentIndex] = JSON.parse(outputText);
                setJsonList(updatedJsonList);

                const newIndex = currentIndex + 1;
                setCurrentIndex(newIndex);
                setOutputText(JSON.stringify(updatedJsonList[newIndex], null, 2));
            } catch (error) {
                alert('Invalid JSON format. Please correct before proceeding.');
                console.error(error);
            }
        }
    };

    const handlePrevious = () => {
        if (currentIndex > 0) {
            try {
                const updatedJsonList = [...jsonList];
                updatedJsonList[currentIndex] = JSON.parse(outputText);
                setJsonList(updatedJsonList);

                const newIndex = currentIndex - 1;
                setCurrentIndex(newIndex);
                setOutputText(JSON.stringify(updatedJsonList[newIndex], null, 2));
            } catch (error) {
                alert('Invalid JSON format. Please correct before proceeding.');
                console.error(error);
            }
        }
    };

    const handleCorrect = async () => {
        try {
            const updatedJsonList = [...jsonList];
            updatedJsonList[currentIndex] = JSON.parse(outputText);
            setJsonList(updatedJsonList);

            const data = await submitCorrection(updatedJsonList, reportId);
            console.log(data);

            // Automatically set next report type
            if (reportType === "klinisk") {
                setReportType("makroskopisk");
            }
            else if (reportType === "makroskopisk") {
                setReportType("mikroskopisk");
            }
            else if (reportType === "mikroskopisk") {
                setReportType("diagnose");
            }
            else {
                setReportType("klinisk");
            }
        } catch (error) {
            if (error instanceof SyntaxError) {
                alert('Invalid JSON format. Please correct before proceeding.');
            } else {
                alert('Error submitting correction. Please try again.');
            }
        }
    };

    const handleGetUnlabeled = async () => {
        const unlabeledJson = await getUnlabeled(reportType);

        // If new case/reportId is fetched, then reset json list
        if (reportId !== unlabeledJson.id) {
            setJsonList([]);
        }

        setReportId(unlabeledJson.id);
        setInputText(unlabeledJson.text);

        setReportType(unlabeledJson.report_type);
        if (unlabeledJson.is_diagnose) {
            setReportType("diagnose");
        }
    }

    const handleToggleForm = (checked: boolean) => {
        if (!checked) {
            setUseFormInput(false);
        } else {
            try {
                const updatedJson = JSON.parse(outputText);
                const updatedJsonList = [...jsonList];
                updatedJsonList[currentIndex] = updatedJson;
                setJsonList(updatedJsonList);
                setUseFormInput(true);
            } catch (error) {
                console.error('Invalid JSON format.');
                setUseFormInput(true);
            }
        }
    };

    const handleToggleEnums = (checked: boolean) => {
        setIncludeEnums(checked);
    }

    const handleToggleGenerateStrings = (checked: boolean) => {
        setGenerateStrings(checked);
    }

    const handleField = (index: number, newValue: string | number | boolean | null) => {
        const updatedJsonList = [...jsonList];

        // Replace empty strings with null value.
        newValue = newValue === "" ? null : newValue;

        updatedJsonList[currentIndex].target_json[index].value = newValue;
        setJsonList(updatedJsonList);
        setOutputText(JSON.stringify(updatedJsonList[currentIndex], null, 2));
    };

    const handleOutput = (value: string | undefined) => {
        setOutputText(value || '');
    };

    const handleClearReportId = () => {
        setReportId(null);
    };

    return (
        <>
            <h1 className="app-title">Pathology Report Labeler</h1>

            <ModelPanel
                modelType={modelType}
                onModelTypeChange={handleSetModelType}
                onModelSelectionChange={setSelectedIndex}
                onLoadModel={handleLoadModel}
                onIsTrainedChange={handleIsTrained}
                isTrained={isTrained}
                index={selectedIndex}
                isLoading={isLoading.loadModel}
            />

            <div className="app-container">
                <InputPanel
                    reportType={reportType}
                    totalContainers={totalContainers}
                    inputText={inputText}
                    onReportTypeChange={setReportType}
                    onTotalContainersChange={setTotalContainers}
                    onInputTextChange={setInputText}
                    onGenerate={handleGenerate}
                    onGetUnlabeled={handleGetUnlabeled}
                    isLoading={isLoading.generate}
                    reportId={reportId}
                />

                <OutputPanel
                    reportId={reportId}
                    useFormInput={useFormInput}
                    onToggleFormChange={handleToggleForm}
                    includeEnums={includeEnums}
                    generateStrings={generateStrings}
                    onToggleGenerateStringsChange={handleToggleGenerateStrings}
                    onToggleEnumsChange={handleToggleEnums}
                    outputText={outputText}
                    onOutputChange={handleOutput}
                    jsonList={jsonList}
                    currentItem={jsonList[currentIndex]}
                    totalItems={jsonList.length}
                    onFieldChange={handleField}
                    currentIndex={currentIndex}
                    onPrevious={handlePrevious}
                    onNext={handleNext}
                    onCorrect={handleCorrect}
                    onClearReportId={handleClearReportId}
                    isLoading={isLoading}
                    isDisabled={jsonList.length === 0}
                />
            </div>
        </>
    );
}

export default App; 