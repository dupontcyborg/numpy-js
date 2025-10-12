/**
 * Unit tests for reduction operations with axis support
 */

import { describe, it, expect } from 'vitest';
import { array } from '../../src/core/ndarray';

describe('Reductions with Axis Support', () => {
  describe('sum()', () => {
    describe('without axis (all elements)', () => {
      it('sums all elements in 1D array', () => {
        const arr = array([1, 2, 3, 4, 5]);
        expect(arr.sum()).toBe(15);
      });

      it('sums all elements in 2D array', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(arr.sum()).toBe(21);
      });

      it('sums all elements in 3D array', () => {
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
        expect(arr.sum()).toBe(36);
      });
    });

    describe('with axis=0', () => {
      it('sums along axis 0 for 2D array', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(0) as any;
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([5, 7, 9]);
      });

      it('sums along axis 0 for 3D array', () => {
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
        const result = arr.sum(0) as any;
        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([
          [6, 8],
          [10, 12],
        ]);
      });
    });

    describe('with axis=1', () => {
      it('sums along axis 1 for 2D array', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(1) as any;
        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([6, 15]);
      });

      it('sums along axis 1 for 3D array', () => {
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
        const result = arr.sum(1) as any;
        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([
          [4, 6],
          [12, 14],
        ]);
      });
    });

    describe('with negative axis', () => {
      it('handles axis=-1 (last axis)', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(-1) as any;
        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([6, 15]);
      });

      it('handles axis=-2 (second-to-last axis)', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(-2) as any;
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([5, 7, 9]);
      });
    });

    describe('with keepdims=true', () => {
      it('keeps dimensions for axis=0', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(0, true) as any;
        expect(result.shape).toEqual([1, 3]);
        expect(result.toArray()).toEqual([[5, 7, 9]]);
      });

      it('keeps dimensions for axis=1', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(1, true) as any;
        expect(result.shape).toEqual([2, 1]);
        expect(result.toArray()).toEqual([[6], [15]]);
      });
    });

    describe('error cases', () => {
      it('throws on out of bounds axis', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(() => arr.sum(2)).toThrow(/out of bounds/);
      });

      it('throws on negative out of bounds axis', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(() => arr.sum(-3)).toThrow(/out of bounds/);
      });
    });
  });

  describe('mean()', () => {
    describe('without axis', () => {
      it('computes mean of all elements', () => {
        const arr = array([1, 2, 3, 4, 5]);
        expect(arr.mean()).toBe(3);
      });

      it('computes mean of 2D array', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(arr.mean()).toBe(3.5);
      });
    });

    describe('with axis=0', () => {
      it('computes mean along axis 0', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.mean(0) as any;
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([2.5, 3.5, 4.5]);
      });
    });

    describe('with axis=1', () => {
      it('computes mean along axis 1', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.mean(1) as any;
        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([2, 5]);
      });
    });

    describe('with keepdims=true', () => {
      it('keeps dimensions for axis=0', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.mean(0, true) as any;
        expect(result.shape).toEqual([1, 3]);
        expect(result.toArray()).toEqual([[2.5, 3.5, 4.5]]);
      });
    });
  });

  describe('max()', () => {
    describe('without axis', () => {
      it('finds max of all elements', () => {
        const arr = array([3, 1, 4, 1, 5, 9, 2, 6]);
        expect(arr.max()).toBe(9);
      });

      it('finds max of 2D array', () => {
        const arr = array([
          [1, 5, 3],
          [9, 2, 6],
        ]);
        expect(arr.max()).toBe(9);
      });
    });

    describe('with axis=0', () => {
      it('finds max along axis 0', () => {
        const arr = array([
          [1, 5, 3],
          [9, 2, 6],
        ]);
        const result = arr.max(0) as any;
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([9, 5, 6]);
      });
    });

    describe('with axis=1', () => {
      it('finds max along axis 1', () => {
        const arr = array([
          [1, 5, 3],
          [9, 2, 6],
        ]);
        const result = arr.max(1) as any;
        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([5, 9]);
      });
    });

    describe('with keepdims=true', () => {
      it('keeps dimensions for axis=0', () => {
        const arr = array([
          [1, 5, 3],
          [9, 2, 6],
        ]);
        const result = arr.max(0, true) as any;
        expect(result.shape).toEqual([1, 3]);
        expect(result.toArray()).toEqual([[9, 5, 6]]);
      });
    });
  });

  describe('min()', () => {
    describe('without axis', () => {
      it('finds min of all elements', () => {
        const arr = array([3, 1, 4, 1, 5, 9, 2, 6]);
        expect(arr.min()).toBe(1);
      });

      it('finds min of 2D array', () => {
        const arr = array([
          [5, 3, 8],
          [1, 9, 2],
        ]);
        expect(arr.min()).toBe(1);
      });
    });

    describe('with axis=0', () => {
      it('finds min along axis 0', () => {
        const arr = array([
          [5, 3, 8],
          [1, 9, 2],
        ]);
        const result = arr.min(0) as any;
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([1, 3, 2]);
      });
    });

    describe('with axis=1', () => {
      it('finds min along axis 1', () => {
        const arr = array([
          [5, 3, 8],
          [1, 9, 2],
        ]);
        const result = arr.min(1) as any;
        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([3, 1]);
      });
    });

    describe('with keepdims=true', () => {
      it('keeps dimensions for axis=0', () => {
        const arr = array([
          [5, 3, 8],
          [1, 9, 2],
        ]);
        const result = arr.min(0, true) as any;
        expect(result.shape).toEqual([1, 3]);
        expect(result.toArray()).toEqual([[1, 3, 2]]);
      });
    });
  });

  describe('complex scenarios', () => {
    it('handles large matrices', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);

      const sumAxis0 = arr.sum(0) as any;
      expect(sumAxis0.toArray()).toEqual([15, 18, 21, 24]);

      const sumAxis1 = arr.sum(1) as any;
      expect(sumAxis1.toArray()).toEqual([10, 26, 42]);
    });

    it('handles 3D arrays with different axes', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]); // shape: (2, 2, 2)

      const sumAxis0 = arr.sum(0) as any;
      expect(sumAxis0.shape).toEqual([2, 2]);

      const sumAxis1 = arr.sum(1) as any;
      expect(sumAxis1.shape).toEqual([2, 2]);

      const sumAxis2 = arr.sum(2) as any;
      expect(sumAxis2.shape).toEqual([2, 2]);
    });

    it('combines reductions', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);

      // Sum along axis 0, then mean of result
      const sumAxis0 = arr.sum(0) as any;
      const meanOfSum = sumAxis0.mean();
      expect(meanOfSum).toBe(7); // [5, 7, 9] -> mean = 7
    });
  });
});
