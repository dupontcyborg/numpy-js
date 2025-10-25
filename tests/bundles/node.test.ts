/**
 * Smoke test for Node.js CJS bundle
 * Tests the actual distributed bundle from dist/
 */

import { describe, test, expect, beforeAll } from 'vitest';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { createRequire } from 'module';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(import.meta.url);

describe('Node.js CJS Bundle Smoke Test', () => {
  let np: any;

  beforeAll(async () => {
    // Import the actual built CJS bundle using require (via createRequire)
    const bundlePath = resolve(__dirname, '../../dist/numpy.node.cjs');
    np = require(bundlePath);
  });

  test('should export main functions', () => {
    expect(np.array).toBeDefined();
    expect(np.zeros).toBeDefined();
    expect(np.ones).toBeDefined();
    expect(np.arange).toBeDefined();
  });

  test('should create arrays', () => {
    const arr = np.array([1, 2, 3, 4]);
    expect(arr.shape).toEqual([4]);
    expect(arr.toArray()).toEqual([1, 2, 3, 4]);
  });

  test('should perform basic math', () => {
    const a = np.array([1, 2, 3]);
    const b = np.array([4, 5, 6]);
    const result = a.add(b); // add is an NDArray method
    expect(result.toArray()).toEqual([5, 7, 9]);
  });

  test('should handle matrix operations', () => {
    const a = np.array([
      [1, 2],
      [3, 4],
    ]);
    const b = np.array([
      [5, 6],
      [7, 8],
    ]);
    const result = a.matmul(b); // matmul is an NDArray method
    expect(result.shape).toEqual([2, 2]);
    expect(result.toArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  test('should create zeros and ones', () => {
    const z = np.zeros([2, 3]);
    expect(z.shape).toEqual([2, 3]);
    expect(z.toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);

    const o = np.ones([3, 2]);
    expect(o.shape).toEqual([3, 2]);
    expect(o.toArray()).toEqual([
      [1, 1],
      [1, 1],
      [1, 1],
    ]);
  });
});
