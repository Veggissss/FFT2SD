import { FormEditorProps, TargetJsonItem } from '../types';
import '../styles/FormEditor.css';

const FormEditor = ({ jsonList, currentJson, onFieldChange }: FormEditorProps) => {
    if (!currentJson || !jsonList) return null;

    const shouldShowField = (item: TargetJsonItem, jsonItems: TargetJsonItem[]) => {
        const id = item.id;

        // Extract relevant values from jsonList
        const getValue = (fieldId: number) => jsonItems.find(j => j.id === fieldId)?.value ?? undefined;

        const consideredRemoved = getValue(105) as boolean | undefined;
        const macroscopicLook = getValue(104) as string | undefined;
        const sampleMaterial = getValue(108) as string | undefined;
        const diagnosis = getValue(109) as string | undefined;
        const infiltrationDepth = getValue(113) as string | undefined;
        const haggittLevel = getValue(114) as string | undefined;

        const isConditionMet = (arr: string[], value: string | undefined) => value === undefined || arr.includes(value ?? '');

        // Diagnosis-based conditions
        const relevantDiagnoses = ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'];
        const extendedDiagnoses = [...relevantDiagnoses, 'M82112', 'M82632', 'M82612', 'M82102', 'M82142', 'M82131', 'M82132'];
        const isDiagnosisRelevant = isConditionMet(relevantDiagnoses, diagnosis);

        // If llm has given the field a value then show it no matter.
        if (item.value !== null) {
            return true;
        }

        switch (id) {
            case 110: // Differensieringsgrad
            case 112: // Lymfekarinnvekst
            case 118: // Venekarinnvekst
            case 113: // Infiltrasjonsdybde
            case 125: // Antall tumor "buds"
            case 126: // Graden av tumor "budding"
                return isDiagnosisRelevant;

            case 111: // Karsinomets største diameter
                return ((isConditionMet(['P13400'], sampleMaterial) && consideredRemoved !== false) ||
                    (isConditionMet(['P13402', 'P13405', 'P13409'], sampleMaterial))) && isDiagnosisRelevant;

            case 114: // Haggit-klassifikasjon
                return ((infiltrationDepth === undefined || infiltrationDepth === 'pT1') &&
                    (isConditionMet(['P13402', 'P13405', 'P13409'], sampleMaterial)) &&
                    (macroscopicLook === undefined || macroscopicLook === 'Is')) &&
                    isDiagnosisRelevant;

            case 120: // Sm-klassifikasjon
            case 127: // Dybde submukosal
                return (infiltrationDepth === undefined || infiltrationDepth === 'pT1') &&
                    (isConditionMet(['P13402', 'P13405', 'P13409'], sampleMaterial)) &&
                    !isConditionMet(['Nivå 1', 'Nivå 2', 'Nivå 3'], haggittLevel) &&
                    isDiagnosisRelevant;

            case 116: // Reseksjonsrender
                return ((isConditionMet(['P13400'], sampleMaterial) && consideredRemoved !== false) ||
                    (isConditionMet(['P13402', 'P13405', 'P13409'], sampleMaterial))) &&
                    isConditionMet(extendedDiagnoses, diagnosis);

            case 122: // Korteste avstand til reseksjonsrand
                return ((isConditionMet(['P13400'], sampleMaterial) && consideredRemoved !== false) ||
                    (isConditionMet(['P13402', 'P13409'], sampleMaterial))) && isDiagnosisRelevant;

            case 115: // Korteste avstand til nærmeste sidereseksjonsrand
            case 121: // Korteste avstand til dyp reseksjonsrand
                return (isConditionMet(['P13405', 'P13409'], sampleMaterial)) && isDiagnosisRelevant;

            default:
                return true;
        }
    };

    // Get the current container ID being edited in form
    const currentContainerId: number | undefined = currentJson.metadata_json?.find((item) => item.field === 'Beholder-ID')?.value as number | undefined;
    if (!currentContainerId) {
        return null;
    }

    // Get a list of all properties from all reporting steps for the current container ID
    // NOTE: This comes with the assumption that the same container ID is refering to the same sample.
    const jsonItems: TargetJsonItem[] = []
    for (let i = 0; i < jsonList.length; i++) {
        const jsonItem = jsonList[i];
        if (jsonItem.metadata_json.find(item => item.field === "Beholder-ID")?.value as number === currentContainerId) {
            jsonItems.push(...jsonItem.target_json);
        }
    }
    return (
        <div className="form-container">
            {currentJson.target_json.map((item, index) => (
                shouldShowField(item, jsonItems) && (
                    <div key={index} className={`form-item ${item.type === 'boolean' ? 'boolean-input' : ''}`}>
                        <label>{item.field}</label>
                        {item.type === 'enum' ? (
                            <select
                                value={item.value as string || ''}
                                onChange={(e) => onFieldChange(index, e.target.value !== '' ? e.target.value : null)}
                            >
                                <option value="">Select an option</option>
                                {item.enum?.map((option, optIndex) => (
                                    <option key={optIndex} value={option.value}>
                                        {option.group ? `[${option.group}] ` : ''}
                                        {option.name || option.value}
                                    </option>
                                ))}
                            </select>
                        ) : item.type === 'int' ? (
                            <input
                                type="number"
                                min={1}
                                value={item.value !== null && item.value !== undefined ? item.value as number : ''}
                                onChange={(e) => onFieldChange(index, e.target.value ? parseInt(e.target.value) : null)}
                                placeholder="Enter a number"
                            />
                        ) : item.type === 'boolean' ? (
                            <input
                                type="checkbox"
                                checked={item.value as boolean || false}
                                onChange={(e) => onFieldChange(index, e.target.checked)}
                            />
                        ) : (
                            <input
                                type="text"
                                value={item.value as string || ''}
                                onChange={(e) => onFieldChange(index, e.target.value)}
                                placeholder="Enter text"
                            />
                        )}
                    </div>
                )
            ))}
        </div>
    );
};

export default FormEditor; 