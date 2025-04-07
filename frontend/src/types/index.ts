export interface TargetJsonItem {
    field: string;
    type: 'string' | 'int' | 'enum' | 'boolean';
    value: string | number | boolean | null;
    enum?: Array<{
        value: string;
        name?: string;
        group?: string;
    }>;
}

export interface LabeledJsonItem {
    id: string,
    report_type: string,
    text: string
}

export interface JsonItem {
    input_text: string;
    target_json: TargetJsonItem[];
    metadata_json: TargetJsonItem[];
}

export interface LoadingState {
    loadModel: boolean;
    generate: boolean;
    correct: boolean;
}

export interface OutputPanelProps {
    reportId: string | null;
    useFormInput: boolean;
    onToggleChange: (checked: boolean) => void;
    outputText: string;
    onOutputChange: (value: string | undefined) => void;
    currentItem: JsonItem | null;
    onFieldChange: (index: number, value: string | number | boolean | null) => void;
    currentIndex: number;
    totalItems: number;
    onPrevious: () => void;
    onNext: () => void;
    onCorrect: () => void;
    onClearReportId: () => void;
    isLoading: LoadingState;
    isDisabled: boolean;
}

export interface FormEditorProps {
    targetJson: TargetJsonItem[];
    onFieldChange: (index: number, value: string | number | boolean | null) => void;
}

export interface InputPanelProps {
    reportId: string | null;
    reportType: string;
    totalContainers: number | null;
    inputText: string;
    onReportTypeChange: (value: string) => void;
    onTotalContainersChange: (value: number | null) => void;
    onInputTextChange: (value: string) => void;
    onGenerate: () => void;
    onGetUnlabeled: () => void;
    isLoading: boolean;
}

export interface ModelPanelProps {
    modelType: string;
    onModelTypeChange: (value: string) => void;
    onLoadModel: () => void;
    isLoading: boolean;
} 