import { InputPanelProps } from '../types';
import '../styles/InputPanel.css';

const InputPanel = ({
    reportType,
    totalContainers,
    inputText,
    onReportTypeChange,
    onTotalContainersChange,
    onInputTextChange,
    onGenerate,
    onGetUnlabeled,
    isLoading
}: InputPanelProps) => {
    return (
        <div className="left-panel">
            <select
                value={reportType}
                onChange={(e) => onReportTypeChange(e.target.value)}
                disabled={isLoading}
            >
                <option value="auto">Report Type (Auto)</option>
                <option value="klinisk">Klinisk</option>
                <option value="makroskopisk">Makroskopisk</option>
                <option value="mikroskopisk">Mikroskopisk</option>
                <option value="diagnose">Diagnose</option>
            </select>
            <button
                onClick={onGetUnlabeled}
                className="action-button"
                disabled={isLoading}
            >
                2. Get Unlabeled Case ({reportType})
            </button>
            <select
                value={totalContainers || ''}
                onChange={(e) => onTotalContainersChange(e.target.value === '' ? null : Number(e.target.value))}
                disabled={isLoading}
            >
                <option value="">Total Containers (Auto)</option>
                {[...Array(10).keys()].map(i => (
                    <option key={i + 1} value={i + 1}>{i + 1}</option>
                ))}
            </select>
            <textarea
                value={inputText}
                onChange={(e) => onInputTextChange(e.target.value)}
                placeholder="Report text..."
                className="input-textarea"
                disabled={isLoading}
            />

            <button
                onClick={onGenerate}
                className="action-button"
                disabled={isLoading}
            >
                {isLoading ? 'Generating...' : '3. Generate'}
            </button>
        </div>
    );
};

export default InputPanel; 