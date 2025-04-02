import { FormEditorProps } from '../types';
import '../styles/FormEditor.css';

const FormEditor = ({ targetJson, onFieldChange }: FormEditorProps) => {
    if (!targetJson) return null;

    const shouldShowField = (item: any, jsonList: any[]) => {
        // Check if field should be shown based on values defined before
        const id = item.id;

        // From klinisk model
        const consideredRemoved: boolean | null | undefined = jsonList.find(j => j.id === 105)?.value;

        // From macroscopic model
        const sampleMaterial: string | undefined = jsonList.find(j => j.id === 108)?.value;

        // Microscopic model (always present for the field cases)
        const diagnosis: string = jsonList.find(j => j.id === 109)?.value;
        const infiltrationDepth: string = jsonList.find(j => j.id === 113)?.value;
        const haggittLevel: string = jsonList.find(j => j.id == 114)?.value;

        switch (id) {
            case 110:
                // Differensieringsgrad
                return ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 111:
                // Karsinomets største diameter
                return (!sampleMaterial || ['P13400'].includes(sampleMaterial) && (consideredRemoved !== false)) ||
                    (!sampleMaterial || ['P13402', 'P13405', 'P13409'].includes(sampleMaterial)) &&
                    ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 112:
                // Lymfekarinnvekst
                return ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 118:
                // Venekarinnvekst
                return ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 113:
                // Infiltrasjonsdybde 
                return ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 114:
                // Haggit-klassifikasjon
                return (infiltrationDepth === 'pT1' && (!sampleMaterial || ['P13402', 'P13405'].includes(sampleMaterial))) ||
                    (!sampleMaterial || ['P13409'].includes(sampleMaterial) && jsonList.find(j => j.id === 104)?.value === 'Is') &&
                    ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 120:
                // Sm-klassifikasjon
                return infiltrationDepth === 'pT1' && (!sampleMaterial || ['P13402', 'P13405', 'P13409'].includes(sampleMaterial)) &&
                    !['Nivå 1', 'Nivå 2', 'Nivå 3'].includes(haggittLevel) &&
                    ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 127:
                // Dybde submukosal
                return infiltrationDepth === 'pT1' && (!sampleMaterial || ['P13402', 'P13405', 'P13409'].includes(sampleMaterial)) &&
                    !['Nivå 1', 'Nivå 2', 'Nivå 3'].includes(haggittLevel) &&
                    ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 116:
                // Reseksjonsrender 
                return (!sampleMaterial || ['P13400'].includes(sampleMaterial) && (consideredRemoved !== false)) ||
                    (!sampleMaterial || ['P13402', 'P13405', 'P13409'].includes(sampleMaterial)) &&
                    ['M82103', 'M82613', 'M82143', 'M82153', 'M81403', 'M82112', 'M82632', 'M82612', 'M82102', 'M82142', 'M82131', 'M82132', 'M82153'].includes(diagnosis);
            case 122:
                // Korteste avstand til reseksjonsrand
                return (!sampleMaterial || ['P13400'].includes(sampleMaterial) && (consideredRemoved !== false)) ||
                    (!sampleMaterial || ['P13402', 'P13409'].includes(sampleMaterial)) &&
                    ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 115:
                // Korteste avstand til nærmeste sidereseksjonsrand
                return (!sampleMaterial || ['P13405', 'P13409'].includes(sampleMaterial)) &&
                    ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 121:
                // Korteste avstand til dyp reseksjonsrand
                return (!sampleMaterial || ['P13405', 'P13409'].includes(sampleMaterial)) &&
                    ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 125:
                // Antall tumor "buds"
                return ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            case 126:
                // Graden av tumor "budding"
                return ['M82103', 'M82613', 'M82143', 'M82153', 'M81403'].includes(diagnosis);
            default:
                // Default no rule applies
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