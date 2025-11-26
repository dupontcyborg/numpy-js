/**
 * Unit tests for linear algebra operations
 */

import { describe, it, expect } from 'vitest';
import { array, dot, trace, transpose, inner, outer, tensordot } from '../../src';

describe('Linear Algebra Operations', () => {
  describe('dot()', () => {
    describe('1D · 1D -> scalar', () => {
      it('computes inner product of two vectors', () => {
        const a = array([1, 2, 3]);
        const b = array([4, 5, 6]);
        const result = dot(a, b);

        expect(result).toBe(1 * 4 + 2 * 5 + 3 * 6); // 32
      });

      it('works with negative numbers', () => {
        const a = array([1, -2, 3]);
        const b = array([-4, 5, 6]);
        const result = dot(a, b);

        expect(result).toBe(1 * -4 + -2 * 5 + 3 * 6); // 4
      });

      it('works with zeros', () => {
        const a = array([1, 0, 3]);
        const b = array([0, 5, 0]);
        const result = dot(a, b);

        expect(result).toBe(0);
      });

      it('throws on shape mismatch', () => {
        const a = array([1, 2, 3]);
        const b = array([4, 5]);

        expect(() => dot(a, b)).toThrow('incompatible shapes');
      });
    });

    describe('2D · 2D -> 2D (matrix multiplication)', () => {
      it('computes 2x2 @ 2x2 matrix multiplication', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [5, 6],
          [7, 8],
        ]);
        const result = dot(a, b) as any;

        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([
          [19, 22],
          [43, 50],
        ]);
      });

      it('delegates to matmul for 2D·2D case', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [5, 6],
          [7, 8],
        ]);

        const dotResult = dot(a, b) as any;
        const matmulResult = a.matmul(b);

        expect(dotResult.toArray()).toEqual(matmulResult.toArray());
      });
    });

    describe('2D · 1D -> 1D (matrix-vector product)', () => {
      it('computes matrix-vector product', () => {
        const a = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const b = array([7, 8, 9]);
        const result = dot(a, b) as any;

        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([
          1 * 7 + 2 * 8 + 3 * 9, // 50
          4 * 7 + 5 * 8 + 6 * 9, // 122
        ]);
      });

      it('works with different sizes', () => {
        const a = array([
          [1, 2],
          [3, 4],
          [5, 6],
        ]);
        const b = array([7, 8]);
        const result = dot(a, b) as any;

        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([
          1 * 7 + 2 * 8, // 23
          3 * 7 + 4 * 8, // 53
          5 * 7 + 6 * 8, // 83
        ]);
      });

      it('throws on shape mismatch', () => {
        const a = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const b = array([7, 8]); // Wrong size

        expect(() => dot(a, b)).toThrow('incompatible shapes');
      });
    });

    describe('1D · 2D -> 1D (vector-matrix product)', () => {
      it('computes vector-matrix product', () => {
        const a = array([1, 2, 3]);
        const b = array([
          [4, 5],
          [6, 7],
          [8, 9],
        ]);
        const result = dot(a, b) as any;

        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([
          1 * 4 + 2 * 6 + 3 * 8, // 40
          1 * 5 + 2 * 7 + 3 * 9, // 46
        ]);
      });

      it('throws on shape mismatch', () => {
        const a = array([1, 2]);
        const b = array([
          [4, 5],
          [6, 7],
          [8, 9],
        ]);

        expect(() => dot(a, b)).toThrow('incompatible shapes');
      });
    });

    describe('dtype handling', () => {
      it('preserves dtype for int32', () => {
        const a = array([1, 2, 3], 'int32');
        const b = array([4, 5, 6], 'int32');
        const result = a.dot(b);

        expect(result).toBe(32);
      });

      it('handles mixed dtypes', () => {
        const a = array([1.5, 2.5], 'float32');
        const b = array([2, 3], 'int16');
        const resultScalar = dot(a, b);

        expect(typeof resultScalar).toBe('number');
        expect(resultScalar).toBeCloseTo(1.5 * 2 + 2.5 * 3, 5);
      });
    });
  });

  describe('trace()', () => {
    it('computes trace of square matrix', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = trace(a);

      expect(result).toBe(1 + 5 + 9); // 15
    });

    it('works with 2x2 matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = trace(a);

      expect(result).toBe(1 + 4); // 5
    });

    it('handles non-square matrices (takes min dimension)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = trace(a);

      expect(result).toBe(1 + 5); // Only 2 diagonal elements
    });

    it('handles tall non-square matrices', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = trace(a);

      expect(result).toBe(1 + 4); // Only 2 diagonal elements
    });

    it('works with identity matrix', () => {
      const a = array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const result = trace(a);

      expect(result).toBe(3);
    });

    it('handles negative numbers', () => {
      const a = array([
        [-1, 2],
        [3, -4],
      ]);
      const result = trace(a);

      expect(result).toBe(-1 + -4); // -5
    });

    it('throws on 1D array', () => {
      const a = array([1, 2, 3]);
      expect(() => trace(a)).toThrow('requires 2D array');
    });

    it('throws on 3D array', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
      ]);
      expect(() => trace(a)).toThrow('requires 2D array');
    });

    it('handles BigInt (int64)', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'int64'
      );
      const result = trace(a);

      expect(typeof result).toBe('bigint');
      expect(result).toBe(BigInt(5));
    });

    it('preserves float64 type', () => {
      const a = array(
        [
          [1.5, 2.3],
          [3.7, 4.9],
        ],
        'float64'
      );
      const result = trace(a);

      expect(typeof result).toBe('number');
      expect(result).toBeCloseTo(1.5 + 4.9, 10);
    });
  });

  describe('transpose()', () => {
    it('transposes 2D array', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = transpose(a);

      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('is same as method call', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);

      const result1 = transpose(a);
      const result2 = a.transpose();

      expect(result1.toArray()).toEqual(result2.toArray());
    });

    it('works with custom axes', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]); // Shape: (2, 2, 2)

      const result = transpose(a, [2, 0, 1]);

      expect(result.shape).toEqual([2, 2, 2]);
      // Axes permutation: (2, 0, 1) means new[i,j,k] = old[j,k,i]
    });

    it('returns a view (shares memory)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = transpose(a);

      expect(result.base).toBe(a);
      expect(result.flags.OWNDATA).toBe(false);
    });
  });

  describe('inner()', () => {
    it('computes inner product for 1D vectors (same as dot)', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = inner(a, b);

      expect(result).toBe(32); // 1*4 + 2*5 + 3*6
      expect(result).toBe(dot(a, b)); // Should match dot for 1D
    });

    it('computes inner product for 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]); // (2, 2)
      const b = array([
        [5, 6],
        [7, 8],
      ]); // (2, 2)
      const result = a.inner(b) as any;

      // Shape should be (2, 2) - contracts last dim of each
      expect(result.shape).toEqual([2, 2]);
      // result[i,j] = sum_k a[i,k] * b[j,k]
      expect(result.toArray()).toEqual([
        [1 * 5 + 2 * 6, 1 * 7 + 2 * 8], // [17, 23]
        [3 * 5 + 4 * 6, 3 * 7 + 4 * 8], // [39, 53]
      ]);
    });

    it('throws on incompatible last dimensions', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);

      expect(() => inner(a, b)).toThrow("don't match");
    });
  });

  describe('outer()', () => {
    it('computes outer product for 1D vectors', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);
      const result = outer(a, b);

      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [4, 5], // 1*4, 1*5
        [8, 10], // 2*4, 2*5
        [12, 15], // 3*4, 3*5
      ]);
    });

    it('flattens 2D inputs', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]); // Will be flattened to [1,2,3,4]
      const b = array([5, 6]);
      const result = a.outer(b);

      expect(result.shape).toEqual([4, 2]);
      const arr = result.toArray() as number[][];
      expect(arr[0]![0]).toBe(1 * 5);
      expect(arr[3]![1]).toBe(4 * 6);
    });

    it('handles scalars (treats as size-1 arrays)', () => {
      const a = array([2]);
      const b = array([3]);
      const result = a.outer(b);

      expect(result.shape).toEqual([1, 1]);
      expect((result.toArray() as number[][])[0]![0]).toBe(6);
    });
  });

  describe('tensordot()', () => {
    it('computes outer product with axes=0', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const result = tensordot(a, b, 0) as any;

      // axes=0 means no contraction (outer product)
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1 * 3, 1 * 4],
        [2 * 3, 2 * 4],
      ]);
    });

    it('validates axes parameter', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);

      expect(() => tensordot(a, b, -1)).toThrow('non-negative');
      expect(() => tensordot(a, b, 5)).toThrow('exceeds');
    });

    it('computes axes=1 contraction (dot product)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]); // (2, 2)
      const b = array([
        [5, 6],
        [7, 8],
      ]); // (2, 2)
      const result = tensordot(a, b, 1) as any;

      // axes=1: contract last 1 axis of a with first 1 of b
      // Same as matmul
      expect(result.shape).toEqual([2, 2]);
    });

    it('computes axes=2 full contraction (scalar)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]); // (2, 2)
      const b = array([
        [5, 6],
        [7, 8],
      ]); // (2, 2)
      const result = tensordot(a, b, 2);

      // axes=2: contract all axes -> scalar
      // sum of element-wise products
      expect(typeof result).toBe('number');
      expect(result).toBe(1 * 5 + 2 * 6 + 3 * 7 + 4 * 8); // 70
    });

    it('computes custom axis pairs', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
      ]); // (1, 2, 2)
      const b = array([[[5, 6]], [[7, 8]]]); // (2, 1, 2)
      const result = tensordot(a, b, [[2], [2]]) as any;

      // Contract axis 2 of a with axis 2 of b
      // Result shape: (1, 2, 2, 1) = (1, 2) + (2, 1)
      expect(result.shape).toEqual([1, 2, 2, 1]);
    });
  });
});
