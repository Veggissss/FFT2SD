import Editor from "@monaco-editor/react";
import FormEditor from "./FormEditor";
import { OutputPanelProps } from '../types';
import '../styles/OutputPanel.css';

const OutputPanel = ({
    reportId,
    useFormInput,
    includeEnums,
    generateStrings,
    onToggleFormChange,
    onToggleEnumsChange,
    onToggleGenerateStringsChange,
    outputText,
    onOutputChange,
    currentItem,
    jsonList,
    onFieldChange,
    currentIndex,
    totalItems,
    onPrevious,
    onNext,
    onCorrect,
    onClearReportId,
    isLoading,
    isDisabled
}: OutputPanelProps) => {
    return (
        <div className="right-panel">
            <div className="editor-toggle">
                <div>
                    <h4>Form Input: </h4>
                    <label className="switch">
                        <input
                            type="checkbox"
                            checked={useFormInput}
                            onChange={(e) => onToggleFormChange(e.target.checked)}
                            disabled={isLoading.correct}
                        />
                        <span className="slider round"></span>
                    </label>
                </div>
                <div className="generation-settings-container">
                    <label className="generation-settings-item">
                        <input
                            type="checkbox"
                            checked={includeEnums}
                            onChange={(e) => onToggleEnumsChange(e.target.checked)}
                            disabled={isLoading.correct}
                        />
                        <span>Include enums</span>
                    </label>
                    <label className="generation-settings-item">
                        <input
                            type="checkbox"
                            checked={generateStrings}
                            onChange={(e) => onToggleGenerateStringsChange(e.target.checked)}
                            disabled={isLoading.correct}
                        />
                        <span>Generate strings</span>
                    </label>
                </div>
                <div>
                    <h4>ID: </h4>
                    {reportId ? (
                        <a
                            onClick={onClearReportId}
                            style={{ cursor: reportId ? 'pointer' : 'default' }}
                            title={reportId ? "Click to clear report ID" : ""}>
                            {reportId} ❌
                        </a>
                    ) : "N/A"}
                </div>
            </div>
            <div className="editor-container">
                {useFormInput ? (
                    <FormEditor
                        jsonList={jsonList}
                        currentJson={currentItem}
                        onFieldChange={onFieldChange}
                    />
                ) : (
                    <Editor
                        height="100%"
                        defaultLanguage="json"
                        value={outputText}
                        onChange={onOutputChange}
                        theme="vs-dark"
                        options={{
                            minimap: { enabled: false },
                            fontSize: 14,
                            wordWrap: 'on',
                            scrollBeyondLastLine: false,
                            automaticLayout: true,
                            lineNumbers: 'on',
                            lineNumbersMinChars: 3,
                            readOnly: isLoading.correct
                        }}
                    />
                )}
            </div>
            <div className="navigation-controls">
                <h4>Report Type: {currentItem?.metadata_json?.find(item => item.field === "Rapport type")?.value}</h4>
                <h4>Container: {currentItem?.metadata_json?.find(item => item.field === "Beholder-ID")?.value ?? ""}</h4>
            </div>
            <div className="navigation-controls">
                <button
                    onClick={onPrevious}
                    disabled={currentIndex === 0 || totalItems === 0 || isLoading.correct}
                >
                    Prev
                </button>
                <span>{totalItems > 0 ? `${currentIndex + 1} / ${totalItems}` : 'No data'}</span>
                <button
                    onClick={onNext}
                    disabled={currentIndex === totalItems - 1 || totalItems === 0 || isLoading.correct}
                >
                    Next
                </button>
            </div>
            <button
                onClick={onCorrect}
                className="action-button"
                disabled={isDisabled || isLoading.correct || totalItems <= 0}
            >
                {isLoading.correct ? 'Submitting...' : '4. Submit Correction'}
            </button>
        </div>
    );
};

export default OutputPanel; 