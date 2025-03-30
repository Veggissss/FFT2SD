import { FormEditorProps } from '../types';
import '../styles/FormEditor.css';
const FormEditor = ({ targetJson, onFieldChange }: FormEditorProps) => {
    if (!targetJson) return null;

    return (
        <div className="form-container">
            {targetJson.map((item, index) => (
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
                                    {option.name || option.value}
                                    {option.group ? ` (${option.group})` : ''}
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
            ))}
        </div>
    );
};

export default FormEditor; 