import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from '../App';
import { server } from '../mocks/server';

// Setup MSW server for test environment
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('App Component', () => {
    it('renders without crashing', () => {
        render(<App />);
        expect(screen.getByText('Pathology Report Labeler')).toBeDefined();
    });

    it('renders main components', () => {
        render(<App />);
        // ModelPanel
        expect(screen.getByText(/1. Load Model/i)).toBeDefined();
        // InputPanel
        expect(screen.getByText(/2. Get Unlabeled Case/i)).toBeDefined();
        expect(screen.getByText(/3. Generate/i)).toBeDefined();

        // OutputPanel
        expect(screen.getByText(/4. Submit Correction/i)).toBeDefined();
    });
});