import { FormEditorProps } from '../types';
import '../styles/FormEditor.css';

const FormEditor = ({ targetJson, onFieldChange }: FormEditorProps) => {
    if (!targetJson) return null;

    const shouldShowField = (item: any, jsonList: any[]) => {
        const id = item.id;

        // Extract relevant values from jsonList
        const getValue = (fieldId: number) => jsonList.find(j => j.id === fieldId)?.value;

        const consideredRemoved: boolean | null | undefined = getValue(105);
        const macroscopicLook: string | undefined = getValue(104)
        const sampleMaterial: string | undefined = getValue(108);
        const diagnosis: string = getValue(109);
        const infiltrationDepth: string = getValue(113);
        const haggittLevel: string = getValue(114);

        const isConditionMet = (arr: string[], value: string | undefined) => value === undefined || arr.includes(value ?? '');

        // Diagnosis-based conditions
        const relevantDiagnoses = ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'];
        const extendedDiagnoses = [...relevantDiagnoses, 'M82112', 'M82632', 'M82612', 'M82102', 'M82142', 'M82131', 'M82132'];
        const isDiagnosisRelevant = isConditionMet(relevantDiagnoses, diagnosis);

        switch (id) {
            case 110: // Differensieringsgrad
            case 112: // Lymfekarinnvekst
            case 118: // Venekarinnvekst
            case 113: // Infiltrasjonsdybde
            case 125: // Antall tumor "buds"
            case 126: // Graden av tumor "budding"
                return isDiagnosisRelevant;

            case 111: // Karsinomets største diameter
                return (isConditionMet(['P13400'], sampleMaterial) && consideredRemoved !== false) ||
                    (isConditionMet(['P13402', 'P13405', 'P13409'], sampleMaterial)) && isDiagnosisRelevant;

            case 114: // Haggit-klassifikasjon
                return (infiltrationDepth === 'pT1' && (isConditionMet(['P13402', 'P13405'], sampleMaterial))) ||
                    (isConditionMet(['P13409'], sampleMaterial) && (macroscopicLook === undefined || macroscopicLook === 'Is')) && isDiagnosisRelevant;

            case 120: // Sm-klassifikasjon
            case 127: // Dybde submukosal
                return infiltrationDepth === 'pT1' &&
                    (isConditionMet(['P13402', 'P13405', 'P13409'], sampleMaterial)) &&
                    !isConditionMet(['Nivå 1', 'Nivå 2', 'Nivå 3'], haggittLevel) &&
                    isDiagnosisRelevant;

            case 116: // Reseksjonsrender
                return (isConditionMet(['P13400'], sampleMaterial) && consideredRemoved !== false) ||
                    (isConditionMet(['P13402', 'P13405', 'P13409'], sampleMaterial)) &&
                    isConditionMet(extendedDiagnoses, diagnosis);

            case 122: // Korteste avstand til reseksjonsrand
                return (isConditionMet(['P13400'], sampleMaterial) && consideredRemoved !== false) ||
                    (isConditionMet(['P13402', 'P13409'], sampleMaterial)) && isDiagnosisRelevant;

            case 115: // Korteste avstand til nærmeste sidereseksjonsrand
            case 121: // Korteste avstand til dyp reseksjonsrand
                return (isConditionMet(['P13405', 'P13409'], sampleMaterial)) && isDiagnosisRelevant;

            default:
                return true;
        }
    };

    return (
        <div className="form-container">
            {targetJson.map((item, index) => (
                shouldShowField(item, targetJson) && (
                    <div key={index} className={`form-item ${item.type === 'boolean' ? 'boolean-input' : ''}`}>
                        <label>{item.field}</label>
                        {item.type === 'enum' ? (
                            <select
                                value={item.value as string || ''}
                                onChange={(e) => onFieldChange(index, e.target.value)}
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
                                value={item.value as number || ''}
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