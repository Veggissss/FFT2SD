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
    const [modelType, setModelType] = useState('encoder');
    const [reportType, setReportType] = useState('auto');
    const [totalContainers, setTotalContainers] = useState<number | null>(null);
    const [jsonList, setJsonList] = useState<JsonItem[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [useFormInput, setUseFormInput] = useState(false);
    const [reportId, setReportId] = useState<string | null>(null)

    // API hooks
    const { isLoading, loadModel, generateReport, submitCorrection, getUnlabeled } = useApi();

    const handleLoadModel = async () => {
        try {
            const data = await loadModel(modelType);
            console.log(data);
        } catch (error) {
            alert('Error loading model. Please try again.');
        }
    };

    const handleGenerate = async () => {
        try {
            const data = await generateReport(inputText, reportType, totalContainers);
            if (reportType === 'auto') {
                setJsonList([]);
            }

            // Filter out same report type
            const filteredJsonList = jsonList.filter(item => item.metadata_json && item.metadata_json[0].value !== reportType);

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

            // only send the current report type list
            const filteredJsonList = updatedJsonList.filter(item => item.metadata_json && item.metadata_json[0].value === reportType);

            const data = await submitCorrection(filteredJsonList, reportId);
            setOutputText(JSON.stringify(data, null, 2));

            // Automatically set next report type
            if (reportType === "klinisk") {
                setReportType("makroskopisk")
            }
            else if (reportType === "makroskopisk") {
                setReportType("mikroskopisk")
            }
            else {
                setReportType("klinisk")
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
    }

    const handleToggleChange = (checked: boolean) => {
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
                alert('Invalid JSON format. Please correct before switching to form mode.');
                return;
            }
        }
    };

    const handleFieldChange = (index: number, newValue: string | number | boolean | null) => {
        const updatedJsonList = [...jsonList];

        // Replace empty strings with null value.
        newValue = newValue === "" ? null : newValue;

        updatedJsonList[currentIndex].target_json[index].value = newValue;
        setJsonList(updatedJsonList);
        setOutputText(JSON.stringify(updatedJsonList[currentIndex], null, 2));
    };

    const handleOutputChange = (value: string | undefined) => {
        setOutputText(value || '');
    };

    return (
        <>
            <h1 className="app-title">Pathology Report Labeler</h1>

            <ModelPanel
                modelType={modelType}
                onModelTypeChange={setModelType}
                onLoadModel={handleLoadModel}
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
                />

                <OutputPanel
                    reportId={reportId}
                    useFormInput={useFormInput}
                    onToggleChange={handleToggleChange}
                    outputText={outputText}
                    onOutputChange={handleOutputChange}
                    currentItem={jsonList[currentIndex]}
                    onFieldChange={handleFieldChange}
                    currentIndex={currentIndex}
                    totalItems={jsonList.length}
                    onPrevious={handlePrevious}
                    onNext={handleNext}
                    onCorrect={handleCorrect}
                    isLoading={isLoading}
                    isDisabled={jsonList.length === 0}
                />
            </div>
        </>
    );
}

export default App; 