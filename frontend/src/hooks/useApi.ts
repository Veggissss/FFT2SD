import { useState } from 'react';
import { JsonItem, LoadingState, ModelsJson, LabeledJsonItem as UnlabeledJsonItem } from '../types';

const useApi = () => {
    const [isLoading, setIsLoading] = useState<LoadingState>({
        loadModel: false,
        generate: false,
        correct: false
    });

    const baseUrl = import.meta.env.VITE_API_BASE_URL ?? '';
    if (baseUrl.length === 0) {
        console.warn("VITE_API_BASE_URL is not defined in .env. Defaulting to http://localhost:5000");
    }
    const apiBaseUrl = baseUrl || "http://localhost:5000";

    const getUnlabeled = async (reportType: string): Promise<UnlabeledJsonItem> => {
        const response = await fetch(`${apiBaseUrl}/unlabeled/${reportType}`, {
            method: 'GET'
        });
        return await response.json();
    }

    const loadModel = async (modelType: string, modelIndex: number, isTrained: boolean): Promise<JsonItem> => {
        setIsLoading(prev => ({ ...prev, loadModel: true }));
        try {
            const response = await fetch(`${apiBaseUrl}/load_model`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_type: modelType, model_index: modelIndex, is_trained: isTrained }),
            });
            return await response.json();
        } finally {
            setIsLoading(prev => ({ ...prev, loadModel: false }));
        }
    };

    const generateReport = async (
        inputText: string,
        reportType: string,
        totalContainers: number | null,
        includeEnums: boolean,
        generateStrings: boolean
    ): Promise<JsonItem[]> => {
        setIsLoading(prev => ({ ...prev, generate: true }));
        try {
            const response = await fetch(`${apiBaseUrl}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    input_text: inputText,
                    report_type: reportType,
                    total_containers: totalContainers,
                    include_enums: includeEnums,
                    generate_strings: generateStrings,
                }),
            });
            return await response.json();
        } finally {
            setIsLoading(prev => ({ ...prev, generate: false }));
        }
    };

    const submitCorrection = async (jsonList: JsonItem[], report_id: string | null): Promise<JsonItem> => {
        setIsLoading(prev => ({ ...prev, correct: true }));
        try {
            // Allow for path/null for no id.
            const response = await fetch(`${apiBaseUrl}/correct/${report_id}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(jsonList),
            });
            return await response.json();
        } finally {
            setIsLoading(prev => ({ ...prev, correct: false }));
        }
    };

    const getModels = async (): Promise<ModelsJson> => {
        try {
            const response = await fetch(`${apiBaseUrl}/models`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            return await response.json();
        } catch (error) {
            console.error('Error fetching models:', error);
            throw error;
        }
    };

    return {
        isLoading,
        loadModel,
        generateReport,
        submitCorrection,
        getUnlabeled,
        getModels,
    };
};

export default useApi; 