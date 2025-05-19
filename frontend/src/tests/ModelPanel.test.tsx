import { describe, it, expect, vi, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { server } from '../mocks/server';
import ModelPanel from '../components/ModelPanel';

describe('ModelPanel Component', () => {
    const mockProps = {
        modelType: 'encoder',
        onIsTrainedChange: vi.fn(),
        onModelTypeChange: vi.fn(),
        onModelSelectionChange: vi.fn(),
        onLoadModel: vi.fn(),
        isTrained: false,
        index: 0,
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

    it('renders model type dropdown', async () => {
        render(<ModelPanel {...mockProps} />);

        // Both dropdowns are rendered with the second one disabled Fetching models
        expect(screen.getAllByRole('combobox').length).toBe(2);
        expect(screen.getByText('Encoder')).toBeDefined();
        expect(screen.getByText('Encoder-Decoder')).toBeDefined();
        expect(screen.getByText('Decoder')).toBeDefined();


        // After GET /models, the placeholder is replaced with the actual model names
        await waitFor(() => expect(screen.queryByRole('option', { name: 'Fetching models' })).toBeNull());
    });

    it('shows placeholder when models are not yet loaded', async () => {
        render(<ModelPanel {...mockProps} />);
        // Verify dropdown exists
        const comboboxes = screen.getAllByRole('combobox');
        expect(comboboxes.length).toBeGreaterThan(0);

        // Specifically check for option element with this text
        expect(screen.getByRole('option', { name: 'Fetching models' })).toBeDefined();
    });
});
