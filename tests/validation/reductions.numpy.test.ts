/**
 * Python NumPy validation tests for reduction operations with axis support
 *
 * These tests run the same operations in both TypeScript and Python NumPy
 * and verify the results match exactly.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Reductions with Axis Support', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  describe('sum() with axis', () => {
    it('validates sum() without axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum()
`);

      expect(result).toBe(npResult.value);
    });

    it('validates sum(axis=0) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum(0);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=1) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=-1) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum(-1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=-1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=0, keepdims=True) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum(0, true);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=1, keepdims=True) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum(1, true);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=1, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=0) for 3D array', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = arr.sum(0);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.sum(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=1) for 3D array', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = arr.sum(1);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.sum(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=2) for 3D array', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = arr.sum(2);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.sum(axis=2)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('mean() with axis', () => {
    it('validates mean() without axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.mean();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.mean()
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates mean(axis=0) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.mean(0);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.mean(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates mean(axis=1) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.mean(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.mean(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates mean(axis=0, keepdims=True) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.mean(0, true);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.mean(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates mean(axis=-1) for 3D array', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = arr.mean(-1);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.mean(axis=-1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('max() with axis', () => {
    it('validates max() without axis', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.max();

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = arr.max()
`);

      expect(result).toBe(npResult.value);
    });

    it('validates max(axis=0) for 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.max(0);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = arr.max(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates max(axis=1) for 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.max(1);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = arr.max(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates max(axis=0, keepdims=True) for 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.max(0, true);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = arr.max(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates max(axis=-1) for 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.max(-1);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = arr.max(axis=-1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates max(axis=1) for 3D array', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = arr.max(1);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.max(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('min() with axis', () => {
    it('validates min() without axis', () => {
      const arr = array([
        [5, 3, 8],
        [1, 9, 2],
      ]);
      const result = arr.min();

      const npResult = runNumPy(`
arr = np.array([[5, 3, 8], [1, 9, 2]])
result = arr.min()
`);

      expect(result).toBe(npResult.value);
    });

    it('validates min(axis=0) for 2D array', () => {
      const arr = array([
        [5, 3, 8],
        [1, 9, 2],
      ]);
      const result = arr.min(0);

      const npResult = runNumPy(`
arr = np.array([[5, 3, 8], [1, 9, 2]])
result = arr.min(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates min(axis=1) for 2D array', () => {
      const arr = array([
        [5, 3, 8],
        [1, 9, 2],
      ]);
      const result = arr.min(1);

      const npResult = runNumPy(`
arr = np.array([[5, 3, 8], [1, 9, 2]])
result = arr.min(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates min(axis=0, keepdims=True) for 2D array', () => {
      const arr = array([
        [5, 3, 8],
        [1, 9, 2],
      ]);
      const result = arr.min(0, true);

      const npResult = runNumPy(`
arr = np.array([[5, 3, 8], [1, 9, 2]])
result = arr.min(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates min(axis=-1) for 2D array', () => {
      const arr = array([
        [5, 3, 8],
        [1, 9, 2],
      ]);
      const result = arr.min(-1);

      const npResult = runNumPy(`
arr = np.array([[5, 3, 8], [1, 9, 2]])
result = arr.min(axis=-1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates min(axis=2) for 3D array', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = arr.min(2);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.min(axis=2)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('complex scenarios', () => {
    it('validates large matrix reduction', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const result = arr.sum(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
result = arr.sum(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates chained reductions', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = (arr.sum(0) as any).mean();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=0).mean()
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates reduction preserving shape', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.max(1, true);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.max(axis=1, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect((result as any).ndim).toBe(2);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });
});
