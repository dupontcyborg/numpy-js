/**
 * Unit tests for comparison operations
 */

import { describe, it, expect } from 'vitest';
import { array } from '../../src/core/ndarray';

describe('Comparison Operations', () => {
  describe('greater()', () => {
    describe('scalar comparisons', () => {
      it('compares 1D array with scalar', () => {
        const arr = array([1, 2, 3, 4, 5]);
        const result = arr.greater(3);
        expect(result.dtype).toBe('uint8');
        expect(result.shape).toEqual([5]);
        expect(Array.from(result.data)).toEqual([0, 0, 0, 1, 1]);
      });

      it('compares 2D array with scalar', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.greater(3);
        expect(result.shape).toEqual([2, 3]);
        expect(Array.from(result.data)).toEqual([0, 0, 0, 1, 1, 1]);
      });
    });

    describe('array comparisons', () => {
      it('compares two 1D arrays', () => {
        const a = array([1, 2, 3]);
        const b = array([2, 2, 2]);
        const result = a.greater(b);
        expect(result.shape).toEqual([3]);
        expect(Array.from(result.data)).toEqual([0, 0, 1]);
      });

      it('compares two 2D arrays', () => {
        const a = array([
          [1, 5],
          [3, 2],
        ]);
        const b = array([
          [2, 4],
          [3, 1],
        ]);
        const result = a.greater(b);
        expect(result.shape).toEqual([2, 2]);
        expect(Array.from(result.data)).toEqual([0, 1, 0, 1]);
      });
    });

    describe('broadcasting', () => {
      it('broadcasts (3, 4) > (4,)', () => {
        const a = array([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ]);
        const b = array([2, 5, 8, 11]);
        const result = a.greater(b);
        expect(result.shape).toEqual([3, 4]);
        // Row 0: [1>2, 2>5, 3>8, 4>11] = [0,0,0,0]
        // Row 1: [5>2, 6>5, 7>8, 8>11] = [1,1,0,0]
        // Row 2: [9>2, 10>5, 11>8, 12>11] = [1,1,1,1]
        expect(Array.from(result.data)).toEqual([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]);
      });

      it('broadcasts (3, 1) > (1, 4)', () => {
        const a = array([[1], [2], [3]]);
        const b = array([[1, 2, 3, 4]]);
        const result = a.greater(b);
        expect(result.shape).toEqual([3, 4]);
      });
    });
  });

  describe('greater_equal()', () => {
    it('compares with scalar', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.greater_equal(3);
      expect(Array.from(result.data)).toEqual([0, 0, 1, 1, 1]);
    });

    it('compares two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([2, 2, 2]);
      const result = a.greater_equal(b);
      expect(Array.from(result.data)).toEqual([0, 1, 1]);
    });
  });

  describe('less()', () => {
    it('compares with scalar', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.less(3);
      expect(Array.from(result.data)).toEqual([1, 1, 0, 0, 0]);
    });

    it('compares two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([2, 2, 2]);
      const result = a.less(b);
      expect(Array.from(result.data)).toEqual([1, 0, 0]);
    });

    it('broadcasts arrays', () => {
      const a = array([[1], [2], [3]]);
      const b = array([2]);
      const result = a.less(b);
      expect(result.shape).toEqual([3, 1]);
      expect(Array.from(result.data)).toEqual([1, 0, 0]);
    });
  });

  describe('less_equal()', () => {
    it('compares with scalar', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.less_equal(3);
      expect(Array.from(result.data)).toEqual([1, 1, 1, 0, 0]);
    });

    it('compares two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([2, 2, 2]);
      const result = a.less_equal(b);
      expect(Array.from(result.data)).toEqual([1, 1, 0]);
    });
  });

  describe('equal()', () => {
    it('compares with scalar', () => {
      const arr = array([1, 2, 3, 2, 1]);
      const result = arr.equal(2);
      expect(Array.from(result.data)).toEqual([0, 1, 0, 1, 0]);
    });

    it('compares two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 4]);
      const result = a.equal(b);
      expect(Array.from(result.data)).toEqual([1, 1, 0]);
    });

    it('compares 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [1, 0],
        [3, 0],
      ]);
      const result = a.equal(b);
      expect(Array.from(result.data)).toEqual([1, 0, 1, 0]);
    });
  });

  describe('not_equal()', () => {
    it('compares with scalar', () => {
      const arr = array([1, 2, 3, 2, 1]);
      const result = arr.not_equal(2);
      expect(Array.from(result.data)).toEqual([1, 0, 1, 0, 1]);
    });

    it('compares two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 4]);
      const result = a.not_equal(b);
      expect(Array.from(result.data)).toEqual([0, 0, 1]);
    });
  });

  describe('result properties', () => {
    it('returns uint8 dtype for all comparisons', () => {
      const arr = array([1, 2, 3]);
      expect(arr.greater(2).dtype).toBe('uint8');
      expect(arr.less(2).dtype).toBe('uint8');
      expect(arr.equal(2).dtype).toBe('uint8');
      expect(arr.not_equal(2).dtype).toBe('uint8');
      expect(arr.greater_equal(2).dtype).toBe('uint8');
      expect(arr.less_equal(2).dtype).toBe('uint8');
    });

    it('preserves shape in comparisons', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.greater(3);
      expect(result.shape).toEqual(arr.shape);
    });
  });

  describe('edge cases', () => {
    it('handles all false result', () => {
      const arr = array([1, 2, 3]);
      const result = arr.greater(10);
      expect(Array.from(result.data)).toEqual([0, 0, 0]);
    });

    it('handles all true result', () => {
      const arr = array([5, 6, 7]);
      const result = arr.greater(0);
      expect(Array.from(result.data)).toEqual([1, 1, 1]);
    });

    it('compares negative numbers', () => {
      const arr = array([-3, -1, 0, 1, 3]);
      const result = arr.greater(0);
      expect(Array.from(result.data)).toEqual([0, 0, 0, 1, 1]);
    });

    it('handles floating point comparisons', () => {
      const arr = array([1.1, 2.2, 3.3]);
      const result = arr.greater(2.0);
      expect(Array.from(result.data)).toEqual([0, 1, 1]);
    });
  });

  describe('broadcasting errors', () => {
    it('throws on incompatible shapes', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([1, 2, 3]);
      expect(() => a.greater(b)).toThrow(/broadcast/);
    });
  });

  describe('chained comparisons', () => {
    it('can chain comparison results', () => {
      const arr = array([1, 2, 3, 4, 5]);
      // Find elements between 2 and 4 (exclusive)
      const gt2 = arr.greater(2);
      const lt4 = arr.less(4);
      // Manually AND the results
      const both = array(Array.from(gt2.data).map((v, i) => (v && lt4.data[i] ? 1 : 0)));
      expect(Array.from(both.data)).toEqual([0, 0, 1, 0, 0]);
    });
  });

  describe('isclose()', () => {
    describe('scalar comparisons', () => {
      it('compares with exact match', () => {
        const arr = array([1.0, 2.0, 3.0]);
        const result = arr.isclose(2.0);
        expect(result.dtype).toBe('uint8');
        expect(Array.from(result.data)).toEqual([0, 1, 0]);
      });

      it('compares with small difference within tolerance', () => {
        const arr = array([1.0, 2.0, 3.0]);
        const result = arr.isclose(2.00001, 1e-5, 1e-8);
        expect(Array.from(result.data)).toEqual([0, 1, 0]);
      });

      it('compares with difference outside tolerance', () => {
        const arr = array([1.0, 2.0, 3.0]);
        const result = arr.isclose(2.1, 1e-5, 1e-8);
        expect(Array.from(result.data)).toEqual([0, 0, 0]);
      });

      it('uses absolute tolerance correctly', () => {
        const arr = array([1e-9, 2e-9]);
        const result = arr.isclose(0, 1e-5, 1e-8);
        // 1e-9 and 2e-9 are both within atol=1e-8 of 0
        expect(Array.from(result.data)).toEqual([1, 1]);
      });

      it('uses relative tolerance correctly', () => {
        const arr = array([1.0, 100.0]);
        const result = arr.isclose(1.001, 0.01, 0);
        // 1.0 vs 1.001: diff=0.001, threshold=0.01*1.001=0.01001 → close
        // 100.0 vs 1.001: diff=98.999, threshold=0.01*1.001=0.01001 → not close
        expect(Array.from(result.data)).toEqual([1, 0]);
      });
    });

    describe('array comparisons', () => {
      it('compares two 1D arrays', () => {
        const a = array([1.0, 2.0, 3.0]);
        const b = array([1.00001, 2.0, 2.99999]);
        const result = a.isclose(b, 1e-4, 1e-8);
        expect(Array.from(result.data)).toEqual([1, 1, 1]);
      });

      it('compares two 2D arrays', () => {
        const a = array([
          [1.0, 2.0],
          [3.0, 4.0],
        ]);
        const b = array([
          [1.0, 2.1],
          [3.0, 4.0],
        ]);
        const result = a.isclose(b, 1e-5, 1e-8);
        expect(Array.from(result.data)).toEqual([1, 0, 1, 1]);
      });

      it('handles floating point precision issues', () => {
        const a = array([0.1 + 0.2]); // Typically 0.30000000000000004
        const b = array([0.3]);
        const result = a.isclose(b);
        expect(Array.from(result.data)).toEqual([1]);
      });
    });

    describe('broadcasting', () => {
      it('broadcasts (2, 3) with (3,)', () => {
        const a = array([
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
        ]);
        const b = array([1.0, 2.0, 3.0]);
        const result = a.isclose(b);
        expect(result.shape).toEqual([2, 3]);
        expect(Array.from(result.data)).toEqual([1, 1, 1, 0, 0, 0]);
      });

      it('broadcasts (3, 1) with (1, 3)', () => {
        const a = array([[1.0], [2.0], [3.0]]);
        const b = array([[1.0, 2.0, 3.0]]);
        const result = a.isclose(b);
        expect(result.shape).toEqual([3, 3]);
      });
    });
  });

  describe('allclose()', () => {
    describe('scalar comparisons', () => {
      it('returns true when all elements are close', () => {
        const arr = array([1.0, 1.00001, 0.99999]);
        expect(arr.allclose(1.0, 1e-4, 1e-8)).toBe(true);
      });

      it('returns false when any element is not close', () => {
        const arr = array([1.0, 1.1, 1.0]);
        expect(arr.allclose(1.0, 1e-5, 1e-8)).toBe(false);
      });

      it('returns true for exact match', () => {
        const arr = array([2.0, 2.0, 2.0]);
        expect(arr.allclose(2.0)).toBe(true);
      });
    });

    describe('array comparisons', () => {
      it('returns true when arrays are close', () => {
        const a = array([1.0, 2.0, 3.0]);
        const b = array([1.00001, 2.0, 2.99999]);
        expect(a.allclose(b, 1e-4, 1e-8)).toBe(true);
      });

      it('returns false when arrays differ', () => {
        const a = array([1.0, 2.0, 3.0]);
        const b = array([1.0, 2.1, 3.0]);
        expect(a.allclose(b, 1e-5, 1e-8)).toBe(false);
      });

      it('handles 2D arrays', () => {
        const a = array([
          [1.0, 2.0],
          [3.0, 4.0],
        ]);
        const b = array([
          [1.0, 2.0],
          [3.0, 4.0],
        ]);
        expect(a.allclose(b)).toBe(true);
      });

      it('handles floating point precision issues', () => {
        const a = array([0.1 + 0.2]);
        const b = array([0.3]);
        expect(a.allclose(b)).toBe(true);
      });
    });

    describe('broadcasting', () => {
      it('works with broadcast-compatible shapes', () => {
        const a = array([
          [1.0, 2.0, 3.0],
          [1.0, 2.0, 3.0],
        ]);
        const b = array([1.0, 2.0, 3.0]);
        expect(a.allclose(b)).toBe(true);
      });

      it('returns false when any broadcast element differs', () => {
        const a = array([
          [1.0, 2.0, 3.0],
          [1.0, 2.1, 3.0],
        ]);
        const b = array([1.0, 2.0, 3.0]);
        expect(a.allclose(b, 1e-5, 1e-8)).toBe(false);
      });
    });

    describe('edge cases', () => {
      it('handles empty arrays', () => {
        const a = array([]);
        const b = array([]);
        expect(a.allclose(b)).toBe(true);
      });

      it('handles single element', () => {
        const a = array([1.0]);
        const b = array([1.00001]);
        expect(a.allclose(b, 1e-4, 1e-8)).toBe(true);
      });
    });
  });
});
