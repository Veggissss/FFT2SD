import { ModelPanelProps } from '../types';
import '../styles/ModelPanel.css';
const ModelPanel = ({ modelType, onModelTypeChange, onLoadModel, isLoading }: ModelPanelProps) => {
    return (
        <div className="model-panel">
            <select
                value={modelType}
                onChange={(e) => onModelTypeChange(e.target.value)}
                disabled={isLoading}
            >
                <option value="encoder">Encoder</option>
                <option value="encoder-decoder">Encoder-Decoder</option>
                <option value="decoder">Decoder</option>
            </select>
            <button
                onClick={onLoadModel}
                className="action-button"
                disabled={isLoading}
            >
                {isLoading ? 'Loading...' : '1. Load Model'}
            </button>
        </div>
    );
};

export default ModelPanel; 