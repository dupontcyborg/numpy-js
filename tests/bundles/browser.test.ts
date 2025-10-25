/**
 * Smoke test for Browser IIFE bundle
 * Tests the actual distributed bundle from dist/ in a real browser
 */

import { describe, test, expect, beforeAll } from 'vitest';

declare global {
  interface Window {
    np: any;
  }
}

// Declare np as global for TypeScript
declare const np: any;

describe('Browser IIFE Bundle Smoke Test', () => {
  beforeAll(async () => {
    // Load the IIFE bundle via script tag
    const script = document.createElement('script');
    script.src = '/dist/numpy.browser.js';
    document.head.appendChild(script);

    // Wait for script to load
    await new Promise((resolve) => {
      script.onload = resolve;
    });
  });

  test('should export main functions', async () => {
    // @ts-ignore - np is loaded as a global by the IIFE
    expect(typeof np).toBe('object');
    // @ts-ignore
    expect(typeof np.array).toBe('function');
    // @ts-ignore
    expect(typeof np.zeros).toBe('function');
    // @ts-ignore
    expect(typeof np.ones).toBe('function');
    // @ts-ignore
    expect(typeof np.arange).toBe('function');
  });

  test('should create arrays', () => {
    // @ts-ignore
    const arr = np.array([1, 2, 3, 4]);
    expect(arr.shape).toEqual([4]);
    expect(arr.toArray()).toEqual([1, 2, 3, 4]);
  });

  test('should perform basic math', () => {
    // @ts-ignore
    const a = np.array([1, 2, 3]);
    // @ts-ignore
    const b = np.array([4, 5, 6]);
    // @ts-ignore
    const result = a.add(b); // add is an NDArray method
    expect(result.toArray()).toEqual([5, 7, 9]);
  });

  test('should handle matrix operations', () => {
    // @ts-ignore
    const a = np.array([
      [1, 2],
      [3, 4],
    ]);
    // @ts-ignore
    const b = np.array([
      [5, 6],
      [7, 8],
    ]);
    // @ts-ignore
    const result = a.matmul(b); // matmul is an NDArray method
    expect(result.shape).toEqual([2, 2]);
    expect(result.toArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  test('should create zeros and ones', () => {
    // @ts-ignore
    const z = np.zeros([2, 3]);
    expect(z.shape).toEqual([2, 3]);
    expect(z.toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);

    // @ts-ignore
    const o = np.ones([3, 2]);
    expect(o.shape).toEqual([3, 2]);
    expect(o.toArray()).toEqual([
      [1, 1],
      [1, 1],
      [1, 1],
    ]);
  });
});
