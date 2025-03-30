import { useState } from 'react';
import { JsonItem, LoadingState } from '../types';

const useApi = () => {
    const [isLoading, setIsLoading] = useState<LoadingState>({
        loadModel: false,
        generate: false,
        correct: false
    });

    const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

    const loadModel = async (modelType: string): Promise<JsonItem> => {
        setIsLoading(prev => ({ ...prev, loadModel: true }));
        try {
            const response = await fetch(`${apiBaseUrl}/load_model`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_type: modelType }),
            });
            return await response.json();
        } finally {
            setIsLoading(prev => ({ ...prev, loadModel: false }));
        }
    };

    const generateReport = async (
        inputText: string,
        reportType: string,
        totalContainers: number | null
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
                    total_containers: totalContainers
                }),
            });
            return await response.json();
        } finally {
            setIsLoading(prev => ({ ...prev, generate: false }));
        }
    };

    const submitCorrection = async (jsonList: JsonItem[]): Promise<JsonItem> => {
        setIsLoading(prev => ({ ...prev, correct: true }));
        try {
            const response = await fetch(`${apiBaseUrl}/correct`, {
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

    return {
        isLoading,
        loadModel,
        generateReport,
        submitCorrection
    };
};

export default useApi; 