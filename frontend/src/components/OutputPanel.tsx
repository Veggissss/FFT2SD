import Editor from "@monaco-editor/react";
import FormEditor from "./FormEditor";
import { OutputPanelProps } from '../types';
import '../styles/OutputPanel.css';

const OutputPanel = ({
    useFormInput,
    onToggleChange,
    outputText,
    onOutputChange,
    currentItem,
    onFieldChange,
    currentIndex,
    totalItems,
    onPrevious,
    onNext,
    onCorrect,
    isLoading,
    isDisabled
}: OutputPanelProps) => {
    return (
        <div className="right-panel">
            <div className="editor-toggle">
                <h4>Form Input: </h4>
                <label className="switch">
                    <input
                        type="checkbox"
                        checked={useFormInput}
                        onChange={(e) => onToggleChange(e.target.checked)}
                        disabled={isLoading.correct}
                    />
                    <span className="slider round"></span>
                </label>
            </div>
            <div className="editor-container">
                {useFormInput ? (
                    <FormEditor
                        targetJson={currentItem?.target_json || []}
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
                {isLoading.correct ? 'Submitting...' : '3. Submit Correction'}
            </button>
        </div>
    );
};

export default OutputPanel; 