/**
 * Extended unit tests for reduction operations
 * Tests edge cases, BigInt dtypes, and special scenarios
 */

import { describe, it, expect } from 'vitest';
import { array, zeros } from '../../src/core/ndarray';

describe('Extended reduction tests', () => {
  describe('sum() with BigInt dtypes', () => {
    it('sums int64 arrays', () => {
      const arr = array([1, 2, 3, 4, 5], 'int64');
      expect(arr.sum()).toBe(15);
    });

    it('sums uint64 arrays', () => {
      const arr = array([10, 20, 30], 'uint64');
      expect(arr.sum()).toBe(60);
    });

    it('sums int64 array along axis', () => {
      const arr = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'int64'
      );
      const result = arr.sum(0);
      expect((result as any).dtype).toBe('int64');
      expect((result as any).data[0]).toBe(BigInt(5));
      expect((result as any).data[1]).toBe(BigInt(7));
      expect((result as any).data[2]).toBe(BigInt(9));
    });

    it('sums uint64 array along axis', () => {
      const arr = array(
        [
          [1, 2],
          [3, 4],
        ],
        'uint64'
      );
      const result = arr.sum(1);
      expect((result as any).dtype).toBe('uint64');
      expect((result as any).data[0]).toBe(BigInt(3));
      expect((result as any).data[1]).toBe(BigInt(7));
    });

    it('sums int64 with keepdims', () => {
      const arr = array([[1, 2, 3]], 'int64');
      const result = arr.sum(1, true);
      expect((result as any).shape).toEqual([1, 1]);
      expect((result as any).dtype).toBe('int64');
      expect((result as any).data[0]).toBe(BigInt(6));
    });
  });

  describe('mean() with different dtypes', () => {
    it('computes mean of int32 array', () => {
      const arr = array([1, 2, 3, 4, 5], 'int32');
      expect(arr.mean()).toBe(3);
    });

    it('computes mean of int64 array', () => {
      const arr = array([2, 4, 6, 8], 'int64');
      expect(arr.mean()).toBe(5);
    });

    it('computes mean of uint64 array', () => {
      const arr = array([10, 20, 30], 'uint64');
      expect(arr.mean()).toBe(20);
    });

    it('computes mean along axis for int64', () => {
      const arr = array([[1, 2], [3, 4]], 'int64');
      const result = arr.mean(0);
      expect((result as any).shape).toEqual([2]);
      // Mean promotes to float64 for accurate computation
      expect((result as any).dtype).toBe('float64');
      expect((result as any).toArray()).toEqual([2, 3]);
    });

    it('computes mean with keepdims for int64', () => {
      const arr = array([[2, 4, 6]], 'int64');
      const result = arr.mean(1, true);
      expect((result as any).shape).toEqual([1, 1]);
      // Mean returns float64
      expect((result as any).dtype).toBe('float64');
      expect((result as any).toArray()).toEqual([[4]]);
    });
  });

  describe('max() with different dtypes', () => {
    it('finds max of int32 array', () => {
      const arr = array([1, 5, 3, 9, 2], 'int32');
      expect(arr.max()).toBe(9);
    });

    it('finds max of int64 array', () => {
      const arr = array([10, 50, 30, 90, 20], 'int64');
      expect(arr.max()).toBe(90);
    });

    it('finds max of uint64 array', () => {
      const arr = array([100, 500, 300], 'uint64');
      expect(arr.max()).toBe(500);
    });

    it('finds max along axis for int64', () => {
      const arr = array(
        [
          [1, 5],
          [3, 2],
        ],
        'int64'
      );
      const result = arr.max(0);
      expect((result as any).dtype).toBe('int64');
      expect((result as any).data[0]).toBe(BigInt(3));
      expect((result as any).data[1]).toBe(BigInt(5));
    });

    it('finds max with keepdims for uint64', () => {
      const arr = array([[5, 10, 3]], 'uint64');
      const result = arr.max(1, true);
      expect((result as any).shape).toEqual([1, 1]);
      expect((result as any).data[0]).toBe(BigInt(10));
    });

    it('finds max along different axes', () => {
      const arr = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'int32'
      );
      const result0 = arr.max(0);
      expect((result0 as any).toArray()).toEqual([4, 5, 6]);

      const result1 = arr.max(1);
      expect((result1 as any).toArray()).toEqual([3, 6]);
    });
  });

  describe('min() with different dtypes', () => {
    it('finds min of int32 array', () => {
      const arr = array([5, 1, 9, 2, 7], 'int32');
      expect(arr.min()).toBe(1);
    });

    it('finds min of int64 array', () => {
      const arr = array([50, 10, 90, 20, 70], 'int64');
      expect(arr.min()).toBe(10);
    });

    it('finds min of uint64 array', () => {
      const arr = array([500, 100, 300], 'uint64');
      expect(arr.min()).toBe(100);
    });

    it('finds min along axis for int64', () => {
      const arr = array(
        [
          [5, 1],
          [2, 3],
        ],
        'int64'
      );
      const result = arr.min(0);
      expect((result as any).dtype).toBe('int64');
      expect((result as any).data[0]).toBe(BigInt(2));
      expect((result as any).data[1]).toBe(BigInt(1));
    });

    it('finds min with keepdims for uint64', () => {
      const arr = array([[10, 5, 15]], 'uint64');
      const result = arr.min(1, true);
      expect((result as any).shape).toEqual([1, 1]);
      expect((result as any).data[0]).toBe(BigInt(5));
    });

    it('finds min along different axes', () => {
      const arr = array(
        [
          [6, 5, 4],
          [3, 2, 1],
        ],
        'int32'
      );
      const result0 = arr.min(0);
      expect((result0 as any).toArray()).toEqual([3, 2, 1]);

      const result1 = arr.min(1);
      expect((result1 as any).toArray()).toEqual([4, 1]);
    });
  });

  describe('prod() with different dtypes', () => {
    it('computes product of int32 array', () => {
      const arr = array([2, 3, 4], 'int32');
      expect(arr.prod()).toBe(24);
    });

    it('computes product of int64 array', () => {
      const arr = array([2, 3, 4], 'int64');
      expect(arr.prod()).toBe(24);
    });

    it('computes product of uint64 array', () => {
      const arr = array([5, 6], 'uint64');
      expect(arr.prod()).toBe(30);
    });

    it('computes product along axis for int64', () => {
      const arr = array([[2, 3], [4, 5]], 'int64');
      const result = arr.prod(0);
      expect((result as any).dtype).toBe('int64');
      expect((result as any).data[0]).toBe(BigInt(8));
      expect((result as any).data[1]).toBe(BigInt(15));
    });

    it('computes product with keepdims for uint64', () => {
      const arr = array([[2, 3, 4]], 'uint64');
      const result = arr.prod(1, true);
      expect((result as any).shape).toEqual([1, 1]);
      expect((result as any).data[0]).toBe(BigInt(24));
    });
  });

  describe('Edge cases', () => {
    it('handles empty arrays for sum', () => {
      const arr = zeros([0], 'float64');
      expect(arr.sum()).toBe(0);
    });

    it('handles empty arrays for mean', () => {
      const arr = zeros([0], 'float64');
      // Mean of empty array is NaN
      expect(arr.mean()).toBeNaN();
    });

    it('handles single element arrays', () => {
      const arr = array([42]);
      expect(arr.sum()).toBe(42);
      expect(arr.mean()).toBe(42);
      expect(arr.max()).toBe(42);
      expect(arr.min()).toBe(42);
      expect(arr.prod()).toBe(42);
    });

    it('handles 1-d reductions', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.sum(0);
      // For 1-d arrays, sum along axis 0 returns scalar
      expect(result).toBe(15);
    });

    it('handles negative values in int64', () => {
      const arr = array([-5, -3, -1], 'int64');
      expect(arr.sum()).toBe(-9);
      expect(arr.max()).toBe(-1);
      expect(arr.min()).toBe(-5);
    });

    it('handles mixed positive and negative in int32', () => {
      const arr = array([-5, 10, -3, 8], 'int32');
      expect(arr.sum()).toBe(10);
      expect(arr.mean()).toBe(2.5);
      expect(arr.max()).toBe(10);
      expect(arr.min()).toBe(-5);
    });

    it('sums bool arrays', () => {
      const arr = array([1, 0, 1, 1, 0], 'bool');
      expect(arr.sum()).toBe(3);
    });

    it('computes mean of bool arrays', () => {
      const arr = array([1, 0, 1, 1, 0], 'bool');
      expect(arr.mean()).toBe(0.6);
    });
  });

  describe('3D arrays with different axes', () => {
    it('reduces 3D int64 array along axis 0', () => {
      const arr = array(
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ],
        'int64'
      );
      const result = arr.sum(0);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).dtype).toBe('int64');
    });

    it('reduces 3D int64 array along axis 1', () => {
      const arr = array(
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ],
        'int64'
      );
      const result = arr.sum(1);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).dtype).toBe('int64');
    });

    it('reduces 3D int64 array along axis 2', () => {
      const arr = array(
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ],
        'int64'
      );
      const result = arr.sum(2);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).dtype).toBe('int64');
    });

    it('uses negative axis in 3D arrays', () => {
      const arr = array(
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        'int32'
      );
      const result = arr.sum(-1);
      expect((result as any).shape).toEqual([1, 2]);
    });
  });

  describe('std() and var() operations', () => {
    it('computes std of float64 array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.std();
      expect(result).toBeCloseTo(1.4142, 4);
    });

    it('computes var of float64 array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.var();
      expect(result).toBeCloseTo(2.0, 1);
    });

    it('computes std along axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.std(0);
      expect((result as any).shape).toEqual([3]);
    });

    it('computes var with keepdims', () => {
      const arr = array([[1, 2, 3]]);
      const result = arr.var(1, 0, true) as any;
      expect(result.shape).toEqual([1, 1]);
      // Variance of [1, 2, 3] is approximately 0.667
      const variance = result.toArray()[0][0];
      expect(variance).toBeCloseTo(0.6667, 3);
    });

    it('computes std of int32 array', () => {
      const arr = array([2, 4, 6, 8], 'int32');
      const result = arr.std();
      expect(result).toBeGreaterThan(0);
    });
  });

  describe('Error handling', () => {
    it('throws error for invalid axis', () => {
      const arr = array([[1, 2], [3, 4]]);
      expect(() => arr.sum(5)).toThrow(/out of bounds/);
      expect(() => arr.sum(-5)).toThrow(/out of bounds/);
    });

    it('throws error for invalid axis in 3D', () => {
      const arr = array([[[1, 2]]]);
      expect(() => arr.sum(3)).toThrow(/out of bounds/);
      expect(() => arr.sum(-4)).toThrow(/out of bounds/);
    });
  });
});
