import { ModelPanelProps, ModelsJson } from '../types';
import '../styles/ModelPanel.css';
import { useState } from 'react';
import useApi from '../hooks/useApi';

const ModelPanel = ({ modelType, onModelTypeChange, onModelSelectionChange, onLoadModel, index, isLoading }: ModelPanelProps) => {
    const { getModels } = useApi();
    const [models, setModels] = useState<ModelsJson>();
    const handleGetModels = async () => {
        try {
            setModels(await getModels());
        } catch (error) {
            console.error('Error fetching models:', error);
            return null;
        }
    }

    // Fetch models only once
    if (!models) {
        handleGetModels();
    }
    return (
        <div className="model-panel">
            <select
                value={modelType}
                onChange={(e) => onModelTypeChange(e.target.value)}
                disabled={isLoading}
            >
                <option value="encoder">Encoder</option>
                <option value="encoder_decoder">Encoder-Decoder</option>
                <option value="decoder">Decoder</option>
            </select>
            {models && models[modelType as keyof typeof models] && (
                <select
                    value={index}
                    onChange={(e) => onModelSelectionChange(Number(e.target.value))}
                    disabled={isLoading}
                >
                    {models[modelType as keyof typeof models].map((model, index) => (
                        <option key={index} value={index}>
                            {model}
                        </option>
                    ))}
                </select>
            )}
            {(!models || !models[modelType as keyof typeof models]) && (
                <select disabled className="placeholder-select">
                    <option>Fetching models</option>
                </select>
            )}

            <button
                onClick={() => onLoadModel()}
                className="action-button"
                disabled={isLoading}
            >
                {isLoading ? 'Loading...' : '1. Load Model'}
            </button>
        </div>
    );
};

export default ModelPanel;