import { describe, it, expect, vi, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { server } from '../mocks/server';
import FormEditor from '../components/FormEditor';
import { FormEditorProps, JsonItem } from '../types';

describe('FormEditor Component', () => {
    // Create mock data for tests
    const mockCurrentJson: JsonItem = {
        input_text: "Sample input text 1",
        metadata_json: [{ id: 1, type: "int", field: 'Beholder-ID', value: 1 }],
        target_json: [
            {
                id: 110, field: 'Differensieringsgrad', type: 'enum', value: 'Lavgradig histologisk differensiering', enum: [
                    { value: "Lavgradig histologisk differensiering" }, { value: "Høygradig histologisk differensiering" }, { value: "Differensieringsgrad kan ikke vurderes" }
                ]
            },
            { id: 109, field: 'Diagnose', type: 'enum', value: 'M82103', enum: [{ value: 'M82103', name: 'Adenokarsinom' }] },
            { id: 200, field: 'Text field', type: 'string', value: 'Test text' },
            { id: 201, field: 'Number field', type: 'int', value: 5 },
            { id: 105, field: 'Vurdert komplett fjernet', type: 'boolean', value: true }
        ]
    };
    const mockJsonList: JsonItem[] = [
        {
            input_text: "Sample input text 2",
            metadata_json: [{ id: 1, type: "int", field: 'Beholder-ID', value: 1 }],
            target_json: [
                { id: 105, field: 'Vurdert komplett fjernet', type: 'boolean', value: true }
            ]
        },
        mockCurrentJson,
    ];

    const mockProps: FormEditorProps = {
        currentJson: mockCurrentJson,
        jsonList: mockJsonList,
        onFieldChange: vi.fn()
    };

    // Setup MSW for tests
    beforeAll(() => {
        server.listen();
    });

    afterEach(() => {
        server.resetHandlers();
        vi.clearAllMocks();
    });

    afterAll(() => server.close());

    it('renders form fields when valid props are provided', () => {
        render(<FormEditor {...mockProps} />);

        // Check if form fields are rendered
        expect(screen.getByText('Differensieringsgrad')).not.toBeNull();
        expect(screen.getByText('Diagnose')).not.toBeNull();
        expect(screen.getByText('Text field')).not.toBeNull();
        expect(screen.getByText('Number field')).not.toBeNull();
    });

    it('renders different input types correctly', () => {
        render(<FormEditor {...mockProps} />);

        // Check for select input (enum type)
        const selectInputs = screen.getAllByRole('combobox');
        expect(selectInputs.length).toBeGreaterThan(0);

        // Check for text input
        const textInput = screen.getByDisplayValue('Test text');
        expect(textInput).not.toBeNull();

        // Check for number input
        const numberInput = screen.getByDisplayValue('5');
        expect(numberInput.getAttribute('type')).toBe('number');

        // Check for checkbox
        const checkbox = screen.getByRole('checkbox');
        expect(checkbox).not.toBeNull();
        expect((checkbox as HTMLInputElement).checked).toBe(true);
    });

    it('calls onFieldChange when select input changes', () => {
        render(<FormEditor {...mockProps} />);

        // Find a select input and change it
        const selectInputs = screen.getAllByRole('combobox');
        fireEvent.change(selectInputs[0], { target: { value: 'Differensieringsgrad kan ikke vurderes' } });
        expect(mockProps.onFieldChange).toHaveBeenCalledWith(expect.any(Number), 'Differensieringsgrad kan ikke vurderes');

        fireEvent.change(selectInputs[0], { target: { value: 'TEST_VALUE_NOT_DEFINED' } });
        expect(mockProps.onFieldChange).toHaveBeenCalledWith(expect.any(Number), null);
    });

    it('calls onFieldChange when text input changes', () => {
        render(<FormEditor {...mockProps} />);

        // Find text input and change it
        const textInput = screen.getByDisplayValue('Test text');
        fireEvent.change(textInput, { target: { value: 'New text' } });
        expect(mockProps.onFieldChange).toHaveBeenCalledWith(expect.any(Number), 'New text');
    });

    it('calls onFieldChange when number input changes', () => {
        render(<FormEditor {...mockProps} />);

        // Find number input and change it
        const numberInput = screen.getByDisplayValue('5');
        fireEvent.change(numberInput, { target: { value: '10' } });
        expect(mockProps.onFieldChange).toHaveBeenCalledWith(expect.any(Number), 10);

        fireEvent.change(numberInput, { target: { value: '' } });
        expect(mockProps.onFieldChange).toHaveBeenCalledWith(expect.any(Number), null);
    });

    it('calls onFieldChange when checkbox changes', () => {
        render(<FormEditor {...mockProps} />);

        // Find checkbox and change it
        const checkbox = screen.getByLabelText('Vurdert komplett fjernet') as HTMLInputElement;

        // Initial data is true so checkbox should be checked
        expect(checkbox.checked).toBeTruthy();

        // Click should uncheck the checkbox
        fireEvent.click(checkbox);
        expect(mockProps.onFieldChange).toHaveBeenCalledWith(expect.any(Number), false);
    });
});