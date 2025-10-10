import { describe, it, expect } from 'vitest';
import { zeros, ones, array, NDArray } from '../../src/core/ndarray';

describe('NDArray Creation', () => {
  describe('zeros', () => {
    it('creates 1D array of zeros', () => {
      const arr = zeros([5]);
      expect(arr.shape).toEqual([5]);
      expect(arr.ndim).toBe(1);
      expect(arr.size).toBe(5);
      expect(arr.dtype).toBe('float64');
      expect(Array.from(arr.data)).toEqual([0, 0, 0, 0, 0]);
    });

    it('creates 2D array of zeros', () => {
      const arr = zeros([2, 3]);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.ndim).toBe(2);
      expect(arr.size).toBe(6);
      expect(Array.from(arr.data)).toEqual([0, 0, 0, 0, 0, 0]);
    });

    it('creates 3D array of zeros', () => {
      const arr = zeros([2, 3, 4]);
      expect(arr.shape).toEqual([2, 3, 4]);
      expect(arr.ndim).toBe(3);
      expect(arr.size).toBe(24);
    });
  });

  describe('ones', () => {
    it('creates 1D array of ones', () => {
      const arr = ones([3]);
      expect(arr.shape).toEqual([3]);
      expect(Array.from(arr.data)).toEqual([1, 1, 1]);
    });

    it('creates 2D array of ones', () => {
      const arr = ones([2, 2]);
      expect(arr.shape).toEqual([2, 2]);
      expect(Array.from(arr.data)).toEqual([1, 1, 1, 1]);
    });
  });

  describe('array', () => {
    it('creates 1D array from flat array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.shape).toEqual([5]);
      expect(arr.ndim).toBe(1);
      expect(Array.from(arr.data)).toEqual([1, 2, 3, 4, 5]);
    });

    it('creates 2D array from nested arrays', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.ndim).toBe(2);
      expect(arr.size).toBe(6);
      expect(Array.from(arr.data)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('creates 3D array from nested arrays', () => {
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
      expect(arr.shape).toEqual([2, 2, 2]);
      expect(arr.ndim).toBe(3);
      expect(arr.size).toBe(8);
    });
  });
});

describe('NDArray Operations', () => {
  describe('matmul', () => {
    it('multiplies 2x2 matrices', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const B = array([
        [5, 6],
        [7, 8],
      ]);
      const C = A.matmul(B);

      expect(C.shape).toEqual([2, 2]);
      expect(C.toArray()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    it('multiplies 2x3 and 3x2 matrices', () => {
      const A = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const B = array([
        [7, 8],
        [9, 10],
        [11, 12],
      ]);
      const C = A.matmul(B);

      expect(C.shape).toEqual([2, 2]);
      // A @ B = [[1*7+2*9+3*11, 1*8+2*10+3*12],
      //          [4*7+5*9+6*11, 4*8+5*10+6*12]]
      //       = [[58, 64], [139, 154]]
      expect(C.toArray()).toEqual([
        [58, 64],
        [139, 154],
      ]);
    });

    it('throws on shape mismatch', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const B = array([[1, 2, 3]]);

      expect(() => A.matmul(B)).toThrow('matmul shape mismatch');
    });

    it('throws on non-2D arrays', () => {
      const A = array([1, 2, 3]);
      const B = array([4, 5, 6]);

      expect(() => A.matmul(B)).toThrow('matmul requires 2D arrays');
    });
  });

  describe('toArray', () => {
    it('converts 1D array to JavaScript array', () => {
      const arr = array([1, 2, 3]);
      expect(arr.toArray()).toEqual([1, 2, 3]);
    });

    it('converts 2D array to nested JavaScript array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(arr.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });
  });
});

describe('NDArray Properties', () => {
  it('has correct strides for 2D C-order array', () => {
    const arr = zeros([3, 4]);
    // C-order: last dimension varies fastest
    // For [3, 4], strides should be [4, 1] (in elements, not bytes)
    // @stdlib uses byte strides, so [4*8, 1*8] = [32, 8] for float64
    expect(arr.strides).toEqual([4, 1]);
  });

  it('toString returns readable format', () => {
    const arr = zeros([2, 3]);
    const str = arr.toString();
    expect(str).toContain('shape');
    expect(str).toContain('2,3');
    expect(str).toContain('dtype');
  });
});
