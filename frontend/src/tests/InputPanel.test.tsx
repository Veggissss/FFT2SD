import { describe, it, expect, vi, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { server } from '../mocks/server';
import InputPanel from '../components/InputPanel';
import { InputPanelProps } from '../types';

describe('InputPanel Component', () => {
    // Create mock props
    const mockProps: InputPanelProps = {
        reportId: null,
        reportType: 'klinisk',
        totalContainers: 2,
        inputText: 'Sample input text',
        onReportTypeChange: vi.fn(),
        onTotalContainersChange: vi.fn(),
        onInputTextChange: vi.fn(),
        onGenerate: vi.fn(),
        onGetUnlabeled: vi.fn(),
        isLoading: false
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

    it('renders the component with all elements', () => {
        render(<InputPanel {...mockProps} />);

        // Check for basic elements
        const selectElements = screen.getAllByRole('combobox');
        expect(selectElements.length).toBe(2);
        expect(screen.getByText('2. Get Unlabeled Case (klinisk)')).toBeDefined();
        expect(screen.getByPlaceholderText('Report text...')).toBeDefined();
        expect(screen.getByText('3. Generate')).toBeDefined();
    });

    it('calls onReportTypeChange when report type select changes', () => {
        render(<InputPanel {...mockProps} />);

        // Find report type select and change it
        const reportTypeSelect = screen.getAllByRole('combobox')[0];
        fireEvent.change(reportTypeSelect, { target: { value: 'makroskopisk' } });

        expect(mockProps.onReportTypeChange).toHaveBeenCalledWith('makroskopisk');
    });

    it('calls onTotalContainersChange when total containers select changes', () => {
        render(<InputPanel {...mockProps} />);

        // Find total containers select and change it
        const totalContainersSelect = screen.getAllByRole('combobox')[1];
        fireEvent.change(totalContainersSelect, { target: { value: '5' } });

        expect(mockProps.onTotalContainersChange).toHaveBeenCalledWith(5);
    });

    it('calls onInputTextChange when textarea changes', () => {
        render(<InputPanel {...mockProps} />);

        // Find textarea and change it
        const textarea = screen.getByPlaceholderText('Report text...');
        fireEvent.change(textarea, { target: { value: 'New text' } });

        expect(mockProps.onInputTextChange).toHaveBeenCalledWith('New text');
    });

    it('calls onGetUnlabeled when Get Unlabeled button is clicked', () => {
        render(<InputPanel {...mockProps} />);

        // Find button and click it
        const getUnlabeledButton = screen.getByText('2. Get Unlabeled Case (klinisk)');
        fireEvent.click(getUnlabeledButton);

        expect(mockProps.onGetUnlabeled).toHaveBeenCalled();
    });

    it('calls onGenerate when Generate button is clicked', () => {
        render(<InputPanel {...mockProps} />);

        // Find button and click it
        const generateButton = screen.getByText('3. Generate');
        fireEvent.click(generateButton);

        expect(mockProps.onGenerate).toHaveBeenCalled();
    });

    it('disables controls when isLoading is true', () => {
        const loadingProps = { ...mockProps, isLoading: true };
        render(<InputPanel {...loadingProps} />);

        // All interactive elements should be disabled
        const selects = screen.getAllByRole('combobox') as HTMLSelectElement[];
        const textarea = screen.getByPlaceholderText('Report text...') as HTMLTextAreaElement;
        const getUnlabeledButton = screen.getByText('2. Get Unlabeled Case (klinisk)').closest('button') as HTMLButtonElement;
        const generateButton = screen.getByText('Generating...').closest('button') as HTMLButtonElement;

        expect(selects[0].disabled).toBe(true);
        expect(selects[1].disabled).toBe(true);
        expect(getUnlabeledButton.disabled).toBe(true);
        expect(textarea.disabled).toBe(true);
        expect(generateButton.disabled).toBe(true);
    });

    it('disables textarea when reportId is not null', () => {
        const propsWithReportId = { ...mockProps, reportId: '12345' };
        render(<InputPanel {...propsWithReportId} />);

        const textarea = screen.getByPlaceholderText('Report text...') as HTMLTextAreaElement;
        expect(textarea.disabled).toBe(true);
    });

    it('shows correct loading text on Generate button', () => {
        const loadingProps = { ...mockProps, isLoading: true };
        render(<InputPanel {...loadingProps} />);

        expect(screen.getByText('Generating...')).toBeDefined();
    });

    it('sets totalContainers to null when empty option is selected', () => {
        render(<InputPanel {...mockProps} />);

        // Find total containers select and change it to empty
        const totalContainersSelect = screen.getAllByRole('combobox')[1];
        fireEvent.change(totalContainersSelect, { target: { value: '' } });

        expect(mockProps.onTotalContainersChange).toHaveBeenCalledWith(null);
    });
});