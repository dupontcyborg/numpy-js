import { defineWorkspace } from 'vitest/config';
import path from 'path';

export default defineWorkspace([
  // Node.js tests (unit, validation, integration)
  {
    test: {
      name: 'node',
      include: ['tests/unit/**', 'tests/validation/**', 'tests/integration/**'],
      environment: 'node',
    },
  },
  // Node.js CJS bundle test
  {
    test: {
      name: 'bundle-node',
      include: ['tests/bundles/node.test.ts'],
      environment: 'node',
    },
  },
  // ESM bundle test
  {
    test: {
      name: 'bundle-esm',
      include: ['tests/bundles/esm.test.mjs'],
      environment: 'node',
    },
  },
  // Browser IIFE bundle test (runs in real browser)
  {
    test: {
      name: 'bundle-browser',
      include: ['tests/bundles/browser.test.ts'],
      browser: {
        enabled: true,
        name: 'chromium',
        provider: 'playwright',
        headless: true,
      },
    },
  },
]);
