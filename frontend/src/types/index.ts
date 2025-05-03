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

export interface ModelsJson {
    decoder: [string],
    encoder: [string],
    encoder_decoder: [string],
}

export interface LabeledJsonItem {
    id: string,
    is_diagnose: boolean,
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
    onToggleGenerateStringsChange: (checked: boolean) => void;
    onToggleFormChange: (checked: boolean) => void;
    onToggleEnumsChange: (checked: boolean) => void;
    generateStrings: boolean;
    includeEnums: boolean;
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
    onIsTrainedChange: (value: boolean) => void;
    onModelTypeChange: (value: string) => void;
    onModelSelectionChange: (index: number) => void;
    onLoadModel: () => void;
    index: number;
    isTrained: boolean;
    isLoading: boolean;
} 