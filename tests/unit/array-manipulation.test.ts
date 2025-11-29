/**
 * Unit tests for array manipulation functions
 *
 * Tests: swapaxes, moveaxis, concatenate, stack, vstack, hstack, dstack,
 *        split, array_split, vsplit, hsplit, tile, repeat
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  zeros,
  swapaxes,
  moveaxis,
  concatenate,
  stack,
  vstack,
  hstack,
  dstack,
  split,
  array_split,
  vsplit,
  hsplit,
  tile,
  repeat,
} from '../../src';

describe('Array Manipulation', () => {
  // ========================================
  // swapaxes
  // ========================================
  describe('swapaxes', () => {
    it('swaps axes of 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = swapaxes(arr, 0, 1);
      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('swaps axes of 3D array', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]); // (2, 2, 2)
      const result = swapaxes(arr, 0, 2);
      expect(result.shape).toEqual([2, 2, 2]);
      expect(result.toArray()).toEqual([
        [
          [1, 5],
          [3, 7],
        ],
        [
          [2, 6],
          [4, 8],
        ],
      ]);
    });

    it('handles negative axes', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = swapaxes(arr, -2, -1);
      expect(result.shape).toEqual([3, 2]);
    });

    it('same axis returns view', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = swapaxes(arr, 0, 0);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('returns a view (shares data)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = swapaxes(arr, 0, 1);
      expect(result.base).toBe(arr);
    });
  });

  // ========================================
  // moveaxis
  // ========================================
  describe('moveaxis', () => {
    it('moves single axis', () => {
      const arr = zeros([3, 4, 5]);
      const result = moveaxis(arr, 0, -1);
      expect(result.shape).toEqual([4, 5, 3]);
    });

    it('moves multiple axes', () => {
      const arr = zeros([3, 4, 5]);
      const result = moveaxis(arr, [0, 1], [-1, -2]);
      expect(result.shape).toEqual([5, 4, 3]);
    });

    it('handles negative axis', () => {
      const arr = zeros([3, 4, 5]);
      const result = moveaxis(arr, -1, 0);
      expect(result.shape).toEqual([5, 3, 4]);
    });

    it('returns a view', () => {
      const arr = zeros([3, 4, 5]);
      const result = moveaxis(arr, 0, -1);
      expect(result.base).toBe(arr);
    });
  });

  // ========================================
  // concatenate
  // ========================================
  describe('concatenate', () => {
    it('concatenates 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = concatenate([a, b]);
      expect(result.shape).toEqual([6]);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('concatenates 2D arrays along axis 0', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], 0);
      expect(result.shape).toEqual([4, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
    });

    it('concatenates 2D arrays along axis 1', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], 1);
      expect(result.shape).toEqual([2, 4]);
      expect(result.toArray()).toEqual([
        [1, 2, 5, 6],
        [3, 4, 7, 8],
      ]);
    });

    it('handles negative axis', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], -1);
      expect(result.shape).toEqual([2, 4]);
    });

    it('throws error for incompatible shapes', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([[5, 6, 7]]);
      expect(() => concatenate([a, b], 0)).toThrow();
    });
  });

  // ========================================
  // stack
  // ========================================
  describe('stack', () => {
    it('stacks 1D arrays along axis 0', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], 0);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('stacks 1D arrays along axis 1', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], 1);
      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('stacks 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = stack([a, b], 0);
      expect(result.shape).toEqual([2, 2, 2]);
      expect(result.toArray()).toEqual([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
    });

    it('handles negative axis', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], -1);
      expect(result.shape).toEqual([3, 2]);
    });

    it('throws error for different shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);
      expect(() => stack([a, b])).toThrow();
    });
  });

  // ========================================
  // vstack
  // ========================================
  describe('vstack', () => {
    it('stacks 1D arrays vertically', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = vstack([a, b]);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('stacks 2D arrays vertically', () => {
      const a = array([[1, 2, 3]]);
      const b = array([[4, 5, 6]]);
      const result = vstack([a, b]);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });
  });

  // ========================================
  // hstack
  // ========================================
  describe('hstack', () => {
    it('stacks 1D arrays horizontally', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = hstack([a, b]);
      expect(result.shape).toEqual([6]);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('stacks 2D arrays horizontally', () => {
      const a = array([[1], [2]]);
      const b = array([[3], [4]]);
      const result = hstack([a, b]);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });
  });

  // ========================================
  // dstack
  // ========================================
  describe('dstack', () => {
    it('stacks 1D arrays depth-wise', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = dstack([a, b]);
      expect(result.shape).toEqual([1, 3, 2]);
      expect(result.toArray()).toEqual([
        [
          [1, 4],
          [2, 5],
          [3, 6],
        ],
      ]);
    });

    it('stacks 2D arrays depth-wise', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = dstack([a, b]);
      expect(result.shape).toEqual([2, 2, 2]);
      expect(result.toArray()).toEqual([
        [
          [1, 5],
          [2, 6],
        ],
        [
          [3, 7],
          [4, 8],
        ],
      ]);
    });
  });

  // ========================================
  // split
  // ========================================
  describe('split', () => {
    it('splits 1D array into equal parts', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = split(arr, 3);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([1, 2]);
      expect(result[1]!.toArray()).toEqual([3, 4]);
      expect(result[2]!.toArray()).toEqual([5, 6]);
    });

    it('splits at specified indices', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = split(arr, [2, 4]);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([1, 2]);
      expect(result[1]!.toArray()).toEqual([3, 4]);
      expect(result[2]!.toArray()).toEqual([5, 6]);
    });

    it('splits 2D array along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const result = split(arr, 2, 0);
      expect(result.length).toBe(2);
      expect(result[0]!.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
      expect(result[1]!.toArray()).toEqual([
        [5, 6],
        [7, 8],
      ]);
    });

    it('throws error for unequal division', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(() => split(arr, 2)).toThrow();
    });

    it('returns views (shares data)', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = split(arr, 3);
      expect(result[0]!.base).toBe(arr);
    });
  });

  // ========================================
  // array_split
  // ========================================
  describe('array_split', () => {
    it('splits with unequal parts', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = array_split(arr, 3);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([1, 2]);
      expect(result[1]!.toArray()).toEqual([3, 4]);
      expect(result[2]!.toArray()).toEqual([5]);
    });

    it('handles equal division', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = array_split(arr, 3);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([1, 2]);
      expect(result[1]!.toArray()).toEqual([3, 4]);
      expect(result[2]!.toArray()).toEqual([5, 6]);
    });
  });

  // ========================================
  // vsplit
  // ========================================
  describe('vsplit', () => {
    it('splits 2D array vertically', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const result = vsplit(arr, 2);
      expect(result.length).toBe(2);
      expect(result[0]!.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
      expect(result[1]!.toArray()).toEqual([
        [5, 6],
        [7, 8],
      ]);
    });

    it('throws error for 1D array', () => {
      const arr = array([1, 2, 3, 4]);
      expect(() => vsplit(arr, 2)).toThrow();
    });
  });

  // ========================================
  // hsplit
  // ========================================
  describe('hsplit', () => {
    it('splits 1D array horizontally', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = hsplit(arr, 3);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([1, 2]);
      expect(result[1]!.toArray()).toEqual([3, 4]);
      expect(result[2]!.toArray()).toEqual([5, 6]);
    });

    it('splits 2D array horizontally', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = hsplit(arr, 2);
      expect(result.length).toBe(2);
      expect(result[0]!.toArray()).toEqual([
        [1, 2],
        [5, 6],
      ]);
      expect(result[1]!.toArray()).toEqual([
        [3, 4],
        [7, 8],
      ]);
    });
  });

  // ========================================
  // tile
  // ========================================
  describe('tile', () => {
    it('tiles 1D array', () => {
      const arr = array([1, 2, 3]);
      const result = tile(arr, 3);
      expect(result.shape).toEqual([9]);
      expect(result.toArray()).toEqual([1, 2, 3, 1, 2, 3, 1, 2, 3]);
    });

    it('tiles 1D array with 2D reps', () => {
      const arr = array([1, 2]);
      const result = tile(arr, [2, 3]);
      expect(result.shape).toEqual([2, 6]);
      expect(result.toArray()).toEqual([
        [1, 2, 1, 2, 1, 2],
        [1, 2, 1, 2, 1, 2],
      ]);
    });

    it('tiles 2D array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = tile(arr, [2, 2]);
      expect(result.shape).toEqual([4, 4]);
      expect(result.toArray()).toEqual([
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2],
        [3, 4, 3, 4],
      ]);
    });

    it('handles single rep', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = tile(arr, 2);
      expect(result.shape).toEqual([2, 4]);
      expect(result.toArray()).toEqual([
        [1, 2, 1, 2],
        [3, 4, 3, 4],
      ]);
    });
  });

  // ========================================
  // repeat
  // ========================================
  describe('repeat', () => {
    it('repeats elements of 1D array', () => {
      const arr = array([1, 2, 3]);
      const result = repeat(arr, 2);
      expect(result.shape).toEqual([6]);
      expect(result.toArray()).toEqual([1, 1, 2, 2, 3, 3]);
    });

    it('repeats elements of 2D array (flattens)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2);
      expect(result.shape).toEqual([8]);
      expect(result.toArray()).toEqual([1, 1, 2, 2, 3, 3, 4, 4]);
    });

    it('repeats along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2, 0);
      expect(result.shape).toEqual([4, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [1, 2],
        [3, 4],
        [3, 4],
      ]);
    });

    it('repeats along axis 1', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2, 1);
      expect(result.shape).toEqual([2, 4]);
      expect(result.toArray()).toEqual([
        [1, 1, 2, 2],
        [3, 3, 4, 4],
      ]);
    });

    it('repeats with different counts per element', () => {
      const arr = array([1, 2, 3]);
      const result = repeat(arr, [1, 2, 3]);
      expect(result.shape).toEqual([6]);
      expect(result.toArray()).toEqual([1, 2, 2, 3, 3, 3]);
    });
  });
});
