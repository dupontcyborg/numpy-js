/**
 * Unit tests for advanced array functions
 *
 * Tests: broadcast_to, broadcast_arrays, take, put, choose, array_equal
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  zeros,
  ones,
  arange,
  broadcast_to,
  broadcast_arrays,
  take,
  put,
  choose,
  array_equal,
} from '../../src';

describe('Advanced Functions', () => {
  // ========================================
  // broadcast_to
  // ========================================
  describe('broadcast_to', () => {
    it('broadcasts 1D array to 2D', () => {
      const arr = array([1, 2, 3]);
      const result = broadcast_to(arr, [3, 3]);
      expect(result.shape).toEqual([3, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
      ]);
    });

    it('broadcasts scalar-like array', () => {
      const arr = array([5]);
      const result = broadcast_to(arr, [3, 4]);
      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [5, 5, 5, 5],
        [5, 5, 5, 5],
        [5, 5, 5, 5],
      ]);
    });

    it('broadcasts column vector', () => {
      const arr = array([[1], [2], [3]]);
      const result = broadcast_to(arr, [3, 4]);
      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
      ]);
    });

    it('broadcasts row vector', () => {
      const arr = array([[1, 2, 3, 4]]);
      const result = broadcast_to(arr, [3, 4]);
      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
      ]);
    });

    it('returns a view', () => {
      const arr = array([1, 2, 3]);
      const result = broadcast_to(arr, [2, 3]);
      expect(result.base).toBe(arr);
    });

    it('throws error for incompatible shapes', () => {
      const arr = array([1, 2, 3]);
      expect(() => broadcast_to(arr, [2, 4])).toThrow();
    });

    it('throws error when target has fewer dimensions', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => broadcast_to(arr, [4])).toThrow();
    });
  });

  // ========================================
  // broadcast_arrays
  // ========================================
  describe('broadcast_arrays', () => {
    it('broadcasts multiple arrays to common shape', () => {
      const a = array([1, 2, 3]);
      const b = array([[1], [2], [3]]);
      const [ra, rb] = broadcast_arrays(a, b);
      expect(ra.shape).toEqual([3, 3]);
      expect(rb.shape).toEqual([3, 3]);
      expect(ra.toArray()).toEqual([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
      ]);
      expect(rb.toArray()).toEqual([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
      ]);
    });

    it('handles single array', () => {
      const arr = array([1, 2, 3]);
      const [result] = broadcast_arrays(arr);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('handles arrays with same shape', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const [ra, rb] = broadcast_arrays(a, b);
      expect(ra.shape).toEqual([2, 2]);
      expect(rb.shape).toEqual([2, 2]);
    });

    it('broadcasts three arrays', () => {
      const a = array([1, 2, 3, 4]);
      const b = array([[1], [2]]);
      const c = array([[[1]]]);
      const [ra, rb, rc] = broadcast_arrays(a, b, c);
      expect(ra.shape).toEqual([1, 2, 4]);
      expect(rb.shape).toEqual([1, 2, 4]);
      expect(rc.shape).toEqual([1, 2, 4]);
    });

    it('throws error for incompatible shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2]);
      expect(() => broadcast_arrays(a, b)).toThrow();
    });
  });

  // ========================================
  // take
  // ========================================
  describe('take', () => {
    it('takes elements from flattened array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = take(arr, [0, 2, 3]);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 3, 4]);
    });

    it('takes elements along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = take(arr, [0, 2], 0);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [5, 6],
      ]);
    });

    it('takes elements along axis 1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = take(arr, [0, 2], 1);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [4, 6],
      ]);
    });

    it('handles negative indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = take(arr, [-1, -2]);
      expect(result.toArray()).toEqual([5, 4]);
    });

    it('handles duplicate indices', () => {
      const arr = array([1, 2, 3]);
      const result = take(arr, [0, 0, 1, 1]);
      expect(result.toArray()).toEqual([1, 1, 2, 2]);
    });

    it('throws error for out-of-bounds index', () => {
      const arr = array([1, 2, 3]);
      expect(() => take(arr, [10])).toThrow();
    });
  });

  // ========================================
  // put
  // ========================================
  describe('put', () => {
    it('puts scalar value at indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      put(arr, [0, 2], 99);
      expect(arr.toArray()).toEqual([99, 2, 99, 4, 5]);
    });

    it('puts array values at indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([10, 20]);
      put(arr, [0, 2], values);
      expect(arr.toArray()).toEqual([10, 2, 20, 4, 5]);
    });

    it('handles negative indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      put(arr, [-1, -2], 0);
      expect(arr.toArray()).toEqual([1, 2, 3, 0, 0]);
    });

    it('cycles values if not enough provided', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([10, 20]);
      put(arr, [0, 1, 2, 3], values);
      expect(arr.toArray()).toEqual([10, 20, 10, 20, 5]);
    });

    it('throws error for out-of-bounds index', () => {
      const arr = array([1, 2, 3]);
      expect(() => put(arr, [10], 5)).toThrow();
    });
  });

  // ========================================
  // choose
  // ========================================
  describe('choose', () => {
    it('chooses from array of choices', () => {
      const choices = [array([1, 2, 3]), array([10, 20, 30]), array([100, 200, 300])];
      const indices = array([0, 1, 2]);
      const result = choose(indices, choices);
      expect(result.toArray()).toEqual([1, 20, 300]);
    });

    it('handles 2D indices', () => {
      const choices = [array([0, 0]), array([1, 1])];
      const indices = array([
        [0, 1],
        [1, 0],
      ]);
      const result = choose(indices, choices);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [0, 1],
        [1, 0],
      ]);
    });

    it('broadcasts choices to index shape', () => {
      const choices = [array([10]), array([20])];
      const indices = array([0, 1, 0, 1]);
      const result = choose(indices, choices);
      expect(result.toArray()).toEqual([10, 20, 10, 20]);
    });

    it('throws error for out-of-bounds choice index', () => {
      const choices = [array([1, 2, 3])];
      const indices = array([0, 1]); // index 1 is out of bounds
      expect(() => choose(indices, choices)).toThrow();
    });
  });

  // ========================================
  // array_equal
  // ========================================
  describe('array_equal', () => {
    it('returns true for equal arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3]);
      expect(array_equal(a, b)).toBe(true);
    });

    it('returns false for different values', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 4]);
      expect(array_equal(a, b)).toBe(false);
    });

    it('returns false for different shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(array_equal(a, b)).toBe(false);
    });

    it('returns false for different sizes', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2]);
      expect(array_equal(a, b)).toBe(false);
    });

    it('handles 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      expect(array_equal(a, b)).toBe(true);
    });

    it('handles NaN values with equal_nan=false', () => {
      const a = array([1, NaN, 3]);
      const b = array([1, NaN, 3]);
      expect(array_equal(a, b, false)).toBe(false);
    });

    it('handles NaN values with equal_nan=true', () => {
      const a = array([1, NaN, 3]);
      const b = array([1, NaN, 3]);
      expect(array_equal(a, b, true)).toBe(true);
    });

    it('handles empty arrays', () => {
      const a = array([]);
      const b = array([]);
      expect(array_equal(a, b)).toBe(true);
    });
  });
});
