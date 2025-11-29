/**
 * Python NumPy validation tests for advanced array operations
 *
 * Tests: broadcast_to, broadcast_arrays, take, put, choose, array_equal
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  broadcast_to,
  broadcast_arrays,
  take,
  put,
  choose,
  array_equal,
} from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Advanced Functions', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  // ========================================
  // broadcast_to
  // ========================================
  describe('broadcast_to()', () => {
    it('validates broadcast_to 1D to 2D', () => {
      const arr = array([1, 2, 3]);
      const result = broadcast_to(arr, [3, 3]);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.broadcast_to(arr, (3, 3))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates broadcast_to scalar-like', () => {
      const arr = array([5]);
      const result = broadcast_to(arr, [3, 4]);

      const npResult = runNumPy(`
arr = np.array([5])
result = np.broadcast_to(arr, (3, 4))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates broadcast_to column vector', () => {
      const arr = array([[1], [2], [3]]);
      const result = broadcast_to(arr, [3, 4]);

      const npResult = runNumPy(`
arr = np.array([[1], [2], [3]])
result = np.broadcast_to(arr, (3, 4))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates broadcast_to row vector', () => {
      const arr = array([[1, 2, 3, 4]]);
      const result = broadcast_to(arr, [3, 4]);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3, 4]])
result = np.broadcast_to(arr, (3, 4))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates broadcast_to 2D to 3D', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = broadcast_to(arr, [2, 2, 2]);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.broadcast_to(arr, (2, 2, 2))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // broadcast_arrays
  // ========================================
  describe('broadcast_arrays()', () => {
    it('validates broadcast_arrays two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([[1], [2], [3]]);
      const [ra, rb] = broadcast_arrays(a, b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
ra, rb = np.broadcast_arrays(a, b)
result = np.array([ra.tolist(), rb.tolist()])
`);

      expect(ra.shape).toEqual([3, 3]);
      expect(rb.shape).toEqual([3, 3]);
      expect(arraysClose(ra.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(rb.toArray(), npResult.value[1])).toBe(true);
    });

    it('validates broadcast_arrays three arrays', () => {
      const a = array([1, 2, 3, 4]);
      const b = array([[1], [2]]);
      const c = array([[[1]]]);
      const [ra, rb, rc] = broadcast_arrays(a, b, c);

      const npResult = runNumPy(`
a = np.array([1, 2, 3, 4])
b = np.array([[1], [2]])
c = np.array([[[1]]])
ra, rb, rc = np.broadcast_arrays(a, b, c)
result = np.array([ra.tolist(), rb.tolist(), rc.tolist()])
`);

      expect(ra.shape).toEqual([1, 2, 4]);
      expect(rb.shape).toEqual([1, 2, 4]);
      expect(rc.shape).toEqual([1, 2, 4]);
      expect(arraysClose(ra.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(rb.toArray(), npResult.value[1])).toBe(true);
      expect(arraysClose(rc.toArray(), npResult.value[2])).toBe(true);
    });
  });

  // ========================================
  // take
  // ========================================
  describe('take()', () => {
    it('validates take from flattened array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = take(arr, [0, 2, 3]);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.take(arr, [0, 2, 3])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates take along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = take(arr, [0, 2], 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4], [5, 6]])
result = np.take(arr, [0, 2], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates take along axis 1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = take(arr, [0, 2], 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.take(arr, [0, 2], axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates take with negative indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = take(arr, [-1, -2]);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.take(arr, [-1, -2])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates take with duplicate indices', () => {
      const arr = array([1, 2, 3]);
      const result = take(arr, [0, 0, 1, 1]);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.take(arr, [0, 0, 1, 1])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // put
  // ========================================
  describe('put()', () => {
    it('validates put scalar value', () => {
      const arr = array([1, 2, 3, 4, 5]);
      put(arr, [0, 2], 99);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
np.put(arr, [0, 2], 99)
result = arr
`);

      expect(arraysClose(arr.toArray(), npResult.value)).toBe(true);
    });

    it('validates put array values', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([10, 20]);
      put(arr, [0, 2], values);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
np.put(arr, [0, 2], [10, 20])
result = arr
`);

      expect(arraysClose(arr.toArray(), npResult.value)).toBe(true);
    });

    it('validates put with negative indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      put(arr, [-1, -2], 0);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
np.put(arr, [-1, -2], 0)
result = arr
`);

      expect(arraysClose(arr.toArray(), npResult.value)).toBe(true);
    });

    it('validates put cycling values', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([10, 20]);
      put(arr, [0, 1, 2, 3], values);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
np.put(arr, [0, 1, 2, 3], [10, 20])
result = arr
`);

      expect(arraysClose(arr.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // choose
  // ========================================
  describe('choose()', () => {
    it('validates choose from array of choices', () => {
      const choices = [array([1, 2, 3]), array([10, 20, 30]), array([100, 200, 300])];
      const indices = array([0, 1, 2]);
      const result = choose(indices, choices);

      const npResult = runNumPy(`
choices = [np.array([1, 2, 3]), np.array([10, 20, 30]), np.array([100, 200, 300])]
indices = np.array([0, 1, 2])
result = np.choose(indices, choices)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates choose with 2D indices', () => {
      const choices = [array([0, 0]), array([1, 1])];
      const indices = array([
        [0, 1],
        [1, 0],
      ]);
      const result = choose(indices, choices);

      const npResult = runNumPy(`
choices = [np.array([0, 0]), np.array([1, 1])]
indices = np.array([[0, 1], [1, 0]])
result = np.choose(indices, choices)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates choose with broadcasting', () => {
      const choices = [array([10]), array([20])];
      const indices = array([0, 1, 0, 1]);
      const result = choose(indices, choices);

      const npResult = runNumPy(`
choices = [np.array([10]), np.array([20])]
indices = np.array([0, 1, 0, 1])
result = np.choose(indices, choices)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // array_equal
  // ========================================
  describe('array_equal()', () => {
    it('validates array_equal for equal arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3]);
      const result = array_equal(a, b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
result = np.array_equal(a, b)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates array_equal for different values', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 4]);
      const result = array_equal(a, b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([1, 2, 4])
result = np.array_equal(a, b)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates array_equal for different shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = array_equal(a, b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])
result = np.array_equal(a, b)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates array_equal 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      const result = array_equal(a, b);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 2], [3, 4]])
result = np.array_equal(a, b)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates array_equal with NaN (equal_nan=False)', () => {
      const a = array([1, NaN, 3]);
      const b = array([1, NaN, 3]);
      const result = array_equal(a, b, false);

      const npResult = runNumPy(`
a = np.array([1, np.nan, 3])
b = np.array([1, np.nan, 3])
result = np.array_equal(a, b, equal_nan=False)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates array_equal with NaN (equal_nan=True)', () => {
      const a = array([1, NaN, 3]);
      const b = array([1, NaN, 3]);
      const result = array_equal(a, b, true);

      const npResult = runNumPy(`
a = np.array([1, np.nan, 3])
b = np.array([1, np.nan, 3])
result = np.array_equal(a, b, equal_nan=True)
`);

      expect(result).toBe(npResult.value);
    });
  });
});
