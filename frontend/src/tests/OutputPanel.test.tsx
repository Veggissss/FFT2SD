import { describe, it, expect, vi, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { server } from '../mocks/server';
import OutputPanel from '../components/OutputPanel';
import { OutputPanelProps, JsonItem } from '../types';

// Mock the monaco editor
vi.mock('@monaco-editor/react', () => ({
    default: () => <div data-testid="monaco-editor">Monaco Editor</div>,
}));

// Mock the FormEditor component
vi.mock('../components/FormEditor', () => ({
    default: () => <div data-testid="form-editor">Form Editor</div>,
}));

describe('OutputPanel Component', () => {
    // Create mock data for tests
    const mockCurrentItem: JsonItem = {
        input_text: "Sample input text 1",
        metadata_json: [
            { id: 1, type: "int", field: 'Beholder-ID', value: 1 },
            { id: 2, type: "string", field: 'Rapport type', value: 'TestReport' }
        ],
        target_json: [
            { id: 105, field: 'Field 1', type: 'boolean', value: true },
            { id: 106, field: 'Field 2', type: 'string', value: 'Test' }
        ]
    };

    const mockJsonList: JsonItem[] = [
        mockCurrentItem,
    ];

    const mockProps: OutputPanelProps = {
        reportId: '12345',
        useFormInput: false,
        includeEnums: true,
        generateStrings: false,
        onToggleFormChange: vi.fn(),
        onToggleEnumsChange: vi.fn(),
        onToggleGenerateStringsChange: vi.fn(),
        outputText: '{"target_json": "json output"}',
        onOutputChange: vi.fn(),
        currentItem: mockCurrentItem,
        jsonList: mockJsonList,
        onFieldChange: vi.fn(),
        currentIndex: 0,
        totalItems: 2,
        onPrevious: vi.fn(),
        onNext: vi.fn(),
        onCorrect: vi.fn(),
        onClearReportId: vi.fn(),
        isLoading: { correct: false, loadModel: false, generate: false },
        isDisabled: false
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
        render(<OutputPanel {...mockProps} />);

        // Check for basic elements
        expect(screen.getByText('Form Input:')).toBeDefined();
        expect(screen.getByText('Include enums')).toBeDefined();
        expect(screen.getByText('Generate strings')).toBeDefined();
        expect(screen.getByText('ID:')).toBeDefined();
        expect(screen.getByText(/12345/)).toBeDefined();

        // Navigation controls
        expect(screen.getByText('Prev')).toBeDefined();
        expect(screen.getByText('1 / 2')).toBeDefined();
        expect(screen.getByText('Next')).toBeDefined();

        // Submit button
        expect(screen.getByText('4. Submit Correction')).toBeDefined();

        // Metadata display
        expect(screen.getByText('Report Type: TestReport')).toBeDefined();
        expect(screen.getByText('Container: 1')).toBeDefined();
    });

    it('toggles between form and editor view', () => {
        const { rerender } = render(<OutputPanel {...mockProps} />);

        // Initially in editor mode (useFormInput = false)
        expect(screen.getByTestId('monaco-editor')).toBeDefined();

        // Change to form mode
        const updatedProps = { ...mockProps, useFormInput: true };
        rerender(<OutputPanel {...updatedProps} />);

        // Now we should be in form mode
        expect(screen.getByTestId('form-editor')).toBeDefined();
    });

    it('calls onToggleFormChange when form toggle is clicked', () => {
        render(<OutputPanel {...mockProps} />);

        const formToggle = screen.getAllByRole('checkbox')[0];
        fireEvent.click(formToggle);

        expect(mockProps.onToggleFormChange).toHaveBeenCalledWith(true);
    });

    it('calls onToggleEnumsChange when include enums toggle is clicked', () => {
        render(<OutputPanel {...mockProps} />);

        const enumsToggle = screen.getAllByRole('checkbox')[1];
        fireEvent.click(enumsToggle);

        expect(mockProps.onToggleEnumsChange).toHaveBeenCalledWith(false);
    });

    it('calls onToggleGenerateStringsChange when generate strings toggle is clicked', () => {
        render(<OutputPanel {...mockProps} />);

        const stringsToggle = screen.getAllByRole('checkbox')[2];
        fireEvent.click(stringsToggle);

        expect(mockProps.onToggleGenerateStringsChange).toHaveBeenCalledWith(true);
    });

    it('calls onClearReportId when report ID is clicked', () => {
        render(<OutputPanel {...mockProps} />);

        const reportId = screen.getByText(/12345/);
        fireEvent.click(reportId);

        expect(mockProps.onClearReportId).toHaveBeenCalled();
    });

    it('calls onCorrect when Submit button is clicked', () => {
        render(<OutputPanel {...mockProps} />);

        const submitButton = screen.getByText('4. Submit Correction');
        fireEvent.click(submitButton);

        expect(mockProps.onCorrect).toHaveBeenCalled();
    });

    it('disables buttons when isLoading.correct is true', () => {
        const loadingProps = { ...mockProps, isLoading: { ...mockProps.isLoading, correct: true } };

        render(<OutputPanel {...loadingProps} />);

        // clickable elements should be disabled
        expect(screen.getByText('Prev').closest('button')?.disabled).toBe(true);
        expect(screen.getByText('Next').closest('button')?.disabled).toBe(true);
        expect(screen.getByText('Submitting...').closest('button')?.disabled).toBe(true);
    });

    it('shows "N/A" when reportId is not provided', () => {
        const propsWithoutId = { ...mockProps, reportId: '' };
        render(<OutputPanel {...propsWithoutId} />);

        expect(screen.getByText('N/A')).toBeDefined();
    });

    it('shows "No data" when totalItems is 0', () => {
        const propsWithNoItems = { ...mockProps, totalItems: 0, currentIndex: 0 };
        render(<OutputPanel {...propsWithNoItems} />);

        expect(screen.getByText('No data')).toBeDefined();
    });
});